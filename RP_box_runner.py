# from https://radprocess.readthedocs.io/en/latest/subboxes.html running radprocess for subboxes

from radprocess.pipeline.Pipeline import Pipeline
from radprocess.plotting import plot
import numpy as np

#sinks_IDs = [65]  	# List of sink IDs to extract subboxes for
id = 53
box_fov = 1000  		# diameter of the subbox in AU
views = ['xy']

pipe = Pipeline()
cfg = pipe.configparams          # a ConfigParams instance
cfg.polaris.nr_threads = 28
archive_prefix = '/scratch/astro/gabriele.columba/'
cfg.dir.ramses_output = archive_prefix + 'ramses_data/output_01440/'
cfg.dir.pipeline_output = archive_prefix + "results/test/"



def create_mixtures( nbin, a_min=1e-8, a_max=2e-3, fractions=[0.8], slope=-3.5, components=['silicate_d03.nk'] ):
	'''
	Create a dictionary of dust mixtures with the desired nbin. 
	fraction: list of mass fractions for each component in the mixture (same len as components).
	components: list of different dust species opacities (nk files) to be used in each mixtures. 
	'''
	mix = {}
	a_grid = np.logspace( np.log10(a_min), np.log10(a_max), nbin + 1, endpoint=True, dtype=np.float32 )
	comp_dir = '/home/PERSONALE/gabriele.columba/POLARIS/input/dust_nk/'	# directory containing the dust opacity files
	
	for i in range(nbin):		# iterate on mixtures
		for c in range(len(components)):		# iterate on components
			mix[ i ] = { c : {		
				'path': comp_dir + components[c],
				'distribution': 'plaw', 	# power-law size distribution
				'fraction': fractions[c], 	# mass fraction 
				'density': 0, 				# using the default density defined in the dust model file
				'amin': a_grid[ i], 		# in [m]
				'amax': a_grid[ i+1], 		# in [m]
				'index': slope, 			# exponent of size distribution
					}
				}
	
	return mix


# Define the AMR fields you want to extract
cfg.amrsource.rho = True
cfg.amrsource.dustratios = False
cfg.amrsource.vel = False       # velocity (optional)
cfg.amrsource.p = True         # pressure (optional)
cfg.amrsource.temp = True      # temperature (requires pressure)

# Simulation parameters
cfg.sim.size_hole_au = 4.0
cfg.sim.facc = 0.1
# cfg.sim.use_ramses_T = False
cfg.sim.dtogas = 0.01		# dust to gas ratio
cfg.nb_dust = 1		# number of dust species IN MHD SIM !


# # Dust materials.    [either the refractive index tables (.nk files) or the cross-sections (*.dat files) in the POLARIS format]

# dustmix = {		# sacha example
# 	0: { # mixture ID (1st mixture)
# 		0: { # component ID (1st component)
# 			'path': '/home/PERSONALE/gabriele.columba/POLARIS/input/dust_nk/silicate_d03.nk',
# 			'distribution': 'plaw', # power-law size distribution
# 			'fraction': .8, 	# 60% mass fraction of 1st mixture
# 			'density': 0, 		# using the default density defined in the dust model file
# 			'amin': 1.0e-08, 	# in m
# 			'amax': 1.0e-04, 	# in m
# 			'index': -3.5 		# MRN
# 		},
# },}

dustmix = create_mixtures( nbin=cfg.nb_dust, a_min=1e-8, a_max=2e-3, fractions=[0.8], slope=-3.5, components=['silicate_d03.nk'] )

cfg.dust.mixtures = dustmix

# Load RAMSES (Step 1, same as basic pipeline)
pipe.load_ramses()      # CANNOT be skipped if injecting ad hoc dust bins !


## Step 2: Extract subboxes
# Extract subboxes around each sink in both POLARIS and RADMC-3D format:

# POLARIS format
pipe.convert_subboxes(
	which_rad= "polaris",
	box_half_width_au= box_fov/2,    # ±500 AU around each sink
	isolation_radius_au= 200,     	# skip sinks closer than 200 AU to a neighbour
	min_cells= 1000,              	# select boxes with a minimum of cell numbers.
	hole_au= cfg.sim.size_hole_au,	# dig hole around sink (this replaces the value set in the Simulation parameters)
	sink_indices= [id-1],
)

# RADMC-3D format (same extraction, different output format)
pipe.convert_subboxes(
	which_rad="radmc",
	box_half_width_au=box_fov/2,
	sink_indices= [id-1],
)

pipe.convert_to_polaris( )

pipe.run_polaris_opacity( dust_mixtures=dustmix )

pipe.prepare_radmc3d_inputs(
	subbox=True,
	nphot=0.5e6,            # thermal 
	nphot_scat=1e6,	        # scattering	
	n_wavelengths=200,      # 200 default 
	wave_min=0.27,          # µm (match the dustkappa range)
	wave_max=30e3,          # µm
	scattering_mode=1,  	# 0: no scattering, 1: isotropic, 2: anisotropic, 3: full scattering (no polarisation)
	scattering_mode_max=1,
	# modified_random_walk=True,	# BUG
	setthreads=28,
)

pipe.run_radmc3d_mctherm( subbox=f'sink_{id :04d}' )

pipe.merge_temperature( subbox=f'sink_{id :04d}' )


# The fov_au parameter sets the field of view in AU. 
# It should match the box_au used for the RADMC-3D regrid so that density and intensity maps are directly comparable. 
# The image is automatically centered on the sink via the POLARIS detector shift.

pipe.render_images(
	dust_mixtures=dustmix,
	npix=512,
	distance_pc=140.0,
	wavelengths_mm=[3],
	views= views,
	subbox=[f'sink_{id :04d}'],
	fov_au=box_fov,     	# diameter
	polaris_binary= '/home/PERSONALE/gabriele.columba/POLARIS/bin/polaris',
	#cleanup_views= True	# ??? 
)


quantity = 'intensity'
for view in views:

	fig = plot.subbox_mosaic(
		pipeline_output= cfg.dir.pipeline_output + "polaris/",
		quantity= quantity,
		view=view,
		wavelength_idx=0,      # first wavelength
		log=True,
		cmap="inferno",
		figsize=(6, 6),
		fontsize=9,
		sink_folders= [f'sink_{id :04d}'] #[f"sink_{id+1 :04d}" for id in sinks_IDs]
	)

	fig.savefig( cfg.dir.pipeline_output + f'{id}_{view}_{quantity}_ISO' + '.pdf', dpi=300, bbox_inches='tight')


# fig = plot.subbox_mosaic(
# 	cfg.dir.pipeline_output + "polaris/",
# 	quantity= "intensity",
# 	view="xy",
# 	wavelength_idx=0,      # first wavelength
# 	log=True,
# 	cmap="inferno",
#     fontsize=9,
# )

# fig.savefig( cfg.dir.pipeline_output + '57_I_xy' + '.pdf', dpi=200, bbox_inches='tight')


# # Optical depth 
# fig = plot.subbox_mosaic( cfg.dir.pipeline_output + "polaris/", quantity="tau", view="xy", log=False, wavelength_idx=0, cmap='turbo', fontsize=9, )

# fig.savefig( cfg.dir.pipeline_output + '50_tau_xy' + '.pdf', dpi=200, bbox_inches='tight')







# # Dust materials.    [either the refractive index tables (.nk files) or the cross-sections (*.dat files) in the POLARIS format]

# dustmix = {		# sacha example
# 	0: { # mixture ID (1st mixture)
# 		0: { # component ID (1st component)
# 			'path': '/home/PERSONALE/gabriele.columba/POLARIS/input/dust_nk/silicate_d03.nk',
# 			'distribution': 'plaw', # power-law size distribution
# 			'fraction': .8, 	# 60% mass fraction of 1st mixture
# 			'density': 0, 		# using the default density defined in the dust model file
# 			'amin': 1.0e-08, 	# in m
# 			'amax': 1.0e-04, 	# in m
# 			'index': -3.5 		# MRN
# 		},
# 	# 	1 :{ # component ID (2nd component)
# 	# 		'path': '/home/PERSONALE/gabriele.columba/POLARIS/input/dust_nk/organics_p94.nk',
# 	# 		'distribution': 'plaw', # MUST be the same for all components within a single mixture
# 	# 		'fraction': 0.4, # 40% mass fraction of 1st mixture (the total should be 1)
# 	# 		'density': 0,
# 	# 		'amin': 5.0e-08,
# 	# 		'amax': 5.0e-06,
# 	# 		'index': -3.0
# 	# 	}
# 	# },
# 	# 1: { # mixture ID (only if the number of dust density distribution columns is greater than 1)
# 	# 	0: { # component ID (a single component)
# 	# 		'path': '/home/PERSONALE/gabriele.columba/POLARIS/input/dust_nk/silicate_d03.nk',
# 	# 		'distribution': 'logn', # log-normal size distribution
# 	# 		'fraction': 1.0, # 100% mass fraction of 2nd mixture
# 	# 		'density': 3300, # kg/m^3
# 	# 		'amin': 1.0e-08,
# 	# 		'amax': 1.0e-06,
# 	# 		'index': [1.0e-07, 0.5]
# 	# 	}
# 	}
# }
