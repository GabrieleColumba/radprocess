"""RADMC-3D continuum image rendering."""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from radprocess.polaris.view_geometry import STANDARD_VIEW_AXES, resolve_view_geometry


def image_to_fits( image_path, distance_pc, coord, fits_path=None, wavelength_index=None):
    """Convert one wavelength of a RADMC-3D ASCII image to a FITS image.

    ``distance_pc`` is the source distance in parsecs and ``coord`` is the
    ICRS sky position of the image center, e.g. ``"03h10m05s -10d05m30s"``.
    For a multi-wavelength image, specify the zero-based ``wavelength_index``.
    The FITS image contains Stokes I in Jy/pixel, without beam convolution.
    The output defaults to ``image.fits`` beside the input and must not exist.

    Returns the path of the written FITS file. Requires the optional
    ``radmc3dPy`` package in addition to the project's Astropy dependency.
    """
    from astropy.coordinates import SkyCoord
    from radmc3dPy import image as radmc_image

    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"RADMC-3D image not found: {image_path}")
    if not np.isfinite(distance_pc) or distance_pc <= 0:
        raise ValueError("distance_pc must be positive and finite.")
    position = SkyCoord(coord, frame="icrs")
    if not position.isscalar:
        raise ValueError("coord must specify one sky position.")

    fits_path = Path(fits_path) if fits_path is not None else image_path.with_suffix(".fits")
    if fits_path.exists():
        raise FileExistsError(f"FITS output already exists: {fits_path}")

    image = radmc_image.readImage( fname=str(image_path))
    if wavelength_index is None:
        if image.nwav != 1:
            raise ValueError("Specify wavelength_index for a multi-wavelength image.")
        wavelength_index = 0
    
    if (not isinstance(wavelength_index, int) or isinstance(wavelength_index, bool)
            or not 0 <= wavelength_index < image.nwav):
        raise ValueError(f"wavelength_index must be between 0 and {image.nwav - 1}.")

    # Override the library's coordinate parsing with Astropy's ICRS result.
    header = {
        "CRVAL1": float(position.ra.deg),
        "CRVAL2": float(position.dec.deg),
        "RADESYS": "ICRS",
        "WAVELEN": float(image.wav[wavelength_index]),
        "DIST_PC": float(distance_pc),
    }
    image.writeFits(
        fname=str(fits_path), dpc=distance_pc, coord="00h00m00s +00d00m00s",
        stokes="I", ifreq=wavelength_index, fitsheadkeys=header,
    )

    return fits_path


def _camera_angles(geometry):
    """Convert the shared image frame to RADMC-3D incl, phi and posang."""
    ex = np.asarray(geometry["ex"], dtype=float)
    ez = np.asarray(geometry["ez"], dtype=float)
    if not np.all(np.isfinite(ex)) or not np.all(np.isfinite(ez)):
        raise ValueError("Image view must have finite axes.")
    incl = np.arccos(np.clip(ez[2], -1.0, 1.0))
    phi = np.arctan2(-ez[0], -ez[1]) if np.hypot(ez[0], ez[1]) > 1e-12 else 0.0
    right = np.array([np.cos(phi), -np.sin(phi), 0.0])
    up = np.array([np.sin(phi) * np.cos(incl),
                   np.cos(phi) * np.cos(incl), np.sin(incl)])
    posang = np.arctan2(-np.dot(ex, up), np.dot(ex, right))
    return tuple(np.degrees([incl, phi, posang]))


def render_images( radmc_dir, image_output_dir, npix, wavelengths_mm, views,
                  custom_views=None, fov_au=None, point_au=None,
                  radmc3d_binary="radmc3d", label="whole"):
    """Render a multi-wavelength RADMC-3D image for each requested view.

    Returns a mapping from view names to directories containing RADMC-3D's
    native image.out (or image.bout) and a render.log. The model directory is
    never used as the working directory, so image generation cannot overwrite
    its files.
    """
    radmc_dir = Path(radmc_dir).resolve()
    image_output_dir = Path(image_output_dir)

    if not isinstance(npix, int) or isinstance(npix, bool) or npix <= 0:
        raise ValueError("npix must be a positive integer.")
    
    wavelengths = np.asarray(wavelengths_mm, dtype=float)

    if wavelengths.ndim != 1 or not wavelengths.size or not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
        raise ValueError("wavelengths_mm must contain positive finite wavelengths.")
    
    if fov_au is not None and (not np.isfinite(fov_au) or fov_au <= 0):
        raise ValueError("fov_au must be positive and finite.")
    
    if point_au is not None:
        point_au = np.asarray(point_au, dtype=float)
        if point_au.shape != (3,) or not np.all(np.isfinite(point_au)):
            raise ValueError("point_au must contain three finite coordinates.")

    required = ["amr_grid.inp", "dust_density.inp", "dustopac.inp",
                "radmc3d.inp", "wavelength_micron.inp"]
    missing = [name for name in required if not (radmc_dir / name).is_file()]
    
    if not any((radmc_dir / name).is_file() for name in
               ("dust_temperature.dat", "dust_temperature.bdat")):
        missing.append("dust_temperature.dat or dust_temperature.bdat")
    if not list(radmc_dir.glob("dustkappa_*.inp")):
        missing.append("dustkappa_*.inp")
    if missing:
        raise FileNotFoundError(f"Missing RADMC-3D inputs in {radmc_dir}: {missing}")

    available = dict(STANDARD_VIEW_AXES)
    
    if custom_views:
        available.update(custom_views)
    if not views:
        raise ValueError("At least one view is required.")
    for name in [label, *views]:
        if not isinstance(name, str) or name in ("", ".", "..") or Path(name).name != name:
            raise ValueError(f"Invalid output name: {name!r}")
    
    geometries = {}
    
    for name in views:
        if name not in available:
            raise ValueError(f"Unknown view {name!r}. Available: {list(available)}")
        geometries[name] = _camera_angles(resolve_view_geometry(name, available[name]))

    binary = os.fspath(radmc3d_binary)
    
    if os.path.dirname(binary):
        binary = str(Path(binary).resolve())
    
    image_dirs = {}
    
    for name, angles in geometries.items():
        
        view_output = image_output_dir / label / name
        view_output.mkdir(parents=True, exist_ok=True)
        
        with tempfile.TemporaryDirectory(prefix="radprocess-image-") as work:
            work = Path(work)
            
            for source in radmc_dir.iterdir():
                if source.is_file() and (source.suffix in (".inp", ".binp")
                                         or source.name in ("dust_temperature.dat", "dust_temperature.bdat")):
                    if source.name != "camera_wavelength_micron.inp":
                        (work / source.name).symlink_to(source)
            
            (work / "camera_wavelength_micron.inp").write_text(
                f"{len(wavelengths)}\n" + "".join(f"{w * 1000:.17g}\n" for w in wavelengths)
            )
            
            incl, phi, posang = angles
            command = [binary, "image", "loadlambda", "npix", str(npix),
                       "incl", str(incl), "phi", str(phi), "posang", str(posang)]
            
            if fov_au is not None:
                command += ["sizeau", str(fov_au)]
            if point_au is not None:
                command += ["pointau", *(str(x) for x in point_au)]
            
            result = subprocess.run(command, cwd=work, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8")
            
            (view_output / "render.log").write_text(result.stdout, encoding="utf-8")
            
            if result.returncode != 0 or "ERROR" in result.stdout or "ABORTED" in result.stdout:
                raise RuntimeError(f"RADMC-3D imaging failed for {name}; see {view_output / 'render.log'}.")
            
            outputs = [work / filename for filename in ("image.out", "image.bout")
                       if (work / filename).is_file()]
            
            if len(outputs) != 1:
                raise RuntimeError(f"RADMC-3D did not produce exactly one image for {name}.")
            
            # Replace only the generated image, leaving other user files intact.
            import shutil
            shutil.copyfile(outputs[0], view_output / outputs[0].name)
        
        image_dirs[name] = view_output
    
    return image_dirs
