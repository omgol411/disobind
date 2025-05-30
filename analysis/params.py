####### Store parameter files anme for each kinda model. #######
######## ------>"May the Force serve u well..." <------#########
################################################################
from typing import Dict

def parameter_files( model_version: int ) -> Dict:
	params = {
		"interaction": get_interaction_params_dict( model_version ),
		"interface": get_inteface_params_dict( model_version )
	}
	return params


def get_interaction_params_dict( model_version: int ) -> Dict:
	if model_version == 23:
		print( "Using v_23 model parameters for interaction prediction..." )
		interaction_param = {
			# v_23 params -------------
			"cg_1": [
				"Epsilon_3_12.1",
				"model_global-[256, 'ln2', True, 1, '']_[0, 6, 0, 2, 'vanilla', '']_['elu', None]_0.0004_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.98__0"
			],
			"cg_5": [
				"Epsilon_3_12.2",
				"model_global-[256, 'ln2', True, 1, '']_[0, 6, 0, 2, 'vanilla', '']_['elu', None]_0.0004_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.92__0"
			],
			"cg_10": [
				"Epsilon_3_12.4",
				"model_global-[256, 'ln2', True, 1, '']_[0, 6, 0, 2, 'vanilla', '']_['elu', None]_0.0004_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.85__0"
			],
		}
	elif model_version == 21:
		print( "Using v_21 model parameters for interaction prediction..." )
		interaction_param = {
			# v_21 params -------------
			"cg_1": [
				"Epsilon_3_10.3",
				"model_global-[256, 'ln2', True, 1, '']_[0, 6, 0, 2, 'vanilla', '']_['elu', None]_0.0004_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.98__0"
			],
			"cg_5": [
				"Epsilon_3_10.6",
				"model_global-[256, 'ln2', True, 1, '']_[0, 6, 0, 2, 'vanilla', '']_['elu', None]_0.0002_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.95__0"
			],
			"cg_10": [
				"Epsilon_3_10.9",
				"model_global-[64, 'ln2', True, 1, '']_[0, 6, 0, 2, 'vanilla', '']_['elu', None]_0.0004_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.98__0"
			],
		}
	elif model_version == 19:
		print( "Using v_19 model parameters for interaction prediction..." )
		interaction_param = {
			# v_19 params -------------
			"cg_1": [
				"Epsilon_3_6.2",
				"model_global-[256, 'ln2', True, 1, '']_[0, 3, 0, 2, 'vanilla', '']_['elu', None]_0.0004_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.98__0"
			],
			"cg_5": [
				"Epsilon_3_6.1",
				"model_global-[128, 'ln2', True, 1, '']_[0, 3, 0, 2, 'vanilla', '']_['elu', None]_0.0002_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.97__0"
			],
			"cg_10": [
				"Epsilon_3_6",
				"model_global-[128, 'ln2', True, 1, '']_[0, 3, 0, 2, 'vanilla', '']_['elu', None]_0.0001_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.97__0"
			],
		}
	else:
		raise ValueError( f"Unsupported model version = {model_version}. (Supported -> 19/21/23)..." )

	return interaction_param


def get_inteface_params_dict( model_version: int ) -> Dict:
	if model_version == 23:
		print( "Using v_23 model parameters for interface prediction..." )
		interface_param = {
			# v_23 params -------------
			"cg_1": [
				"Epsilon_3_13",
				"model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0002_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.97__0"
				
			],
			"cg_5": [
				"Epsilon_3_13.1",
				"model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0002_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.93__0"
			],
			"cg_10": [
				"Epsilon_3_13.2",
				"model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0002_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.93__0"
			]
		}
	elif model_version == 21:
		print( "Using v_21 model parameters for interface prediction..." )
		interface_param = {
			# v_21 params -------------
			"cg_1": [
				"Epsilon_3_14",
				"model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0003_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.98__0"
				# "Epsilon_3_11.1",
				# "model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0002_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.95__0"
				
			],
			"cg_5": [
				"Epsilon_3_14.1",
				"model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0002_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.94__0"
				# "Epsilon_3_11.3",
				# "model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0001_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.92__0"
			],
			"cg_10": [
				"Epsilon_3_14.2",
				"model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0002_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.9__0"
				# "Epsilon_3_11.4",
				# "model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0001_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.89__0"
			]
		}
	elif model_version == 19:
		print( "Using v_19 model parameters for interface prediction..." )
		interface_param = {
			# v_19 params -------------
			"cg_1": [
				"Epsilon_3_15",
				"model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0002_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.98__0"
				# "Epsilon_3_5.2",
				# "model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0002_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.98__0"
			],
			"cg_5": [
				"Epsilon_3_15.1",
				"model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0002_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.91__0"
				# "Epsilon_3_5.1",
				# "model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0001_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.94__0"
			],
			"cg_10": [
				"Epsilon_3_15.2",
				"model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0002_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.88__0"
				# "Epsilon_3_5",
				# "model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0001_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.93__0"
			]
		}
	else:
		raise ValueError( f"Unsupported model version = {model_version}. (Supported -> 19/21/23)..." )

	return interface_param



	# params = {
	# 	"interaction": {
	# 		# Multi output interaction versions with vanilla block.
	# 		# v_23 params -------------
	# 		# "cg_1": [
	# 		# 	"Epsilon_3_12.1",
	# 		# 	"model_global-[256, 'ln2', True, 1, '']_[0, 6, 0, 2, 'vanilla', '']_['elu', None]_0.0004_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.98__0"
	# 		# ],
	# 		# "cg_5": [
	# 		# 	"Epsilon_3_12.2",
	# 		# 	"model_global-[256, 'ln2', True, 1, '']_[0, 6, 0, 2, 'vanilla', '']_['elu', None]_0.0004_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.92__0"
	# 		# 	# "model_global-[128, 'ln2', True, 1, '']_[0, 6, 0, 2, 'vanilla', '']_['elu', None]_0.0004_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.94__0"
	# 		# ],
	# 		# "cg_10": [
	# 		# 	"Epsilon_3_12.4",
	# 		# 	"model_global-[256, 'ln2', True, 1, '']_[0, 6, 0, 2, 'vanilla', '']_['elu', None]_0.0004_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.85__0"
	# 		# 	# "model_global-[128, 'ln2', True, 1, '']_[0, 6, 0, 2, 'vanilla', '']_['elu', None]_0.0003_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.87__0"
	# 		# ],
	# 		# v_21 params -------------
	# 		"cg_1": [
	# 			"Epsilon_3_10.3",
	# 			"model_global-[256, 'ln2', True, 1, '']_[0, 6, 0, 2, 'vanilla', '']_['elu', None]_0.0004_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.98__0"
	# 		],
	# 		"cg_5": [
	# 			"Epsilon_3_10.6",
	# 			"model_global-[256, 'ln2', True, 1, '']_[0, 6, 0, 2, 'vanilla', '']_['elu', None]_0.0002_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.95__0"
	# 			# "Epsilon_3_10.4",
	# 			# "model_global-[128, 'ln2', True, 1, '']_[0, 6, 0, 2, 'vanilla', '']_['elu', None]_0.0004_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.94__0"
	# 		],
	# 		"cg_10": [
	# 			"Epsilon_3_10.9",
	# 			"model_global-[64, 'ln2', True, 1, '']_[0, 6, 0, 2, 'vanilla', '']_['elu', None]_0.0004_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.98__0"
	# 			# "Epsilon_3_10.5",
	# 			# "model_global-[128, 'ln2', True, 1, '']_[0, 6, 0, 2, 'vanilla', '']_['elu', None]_0.0003_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.87__0"
	# 		],
	# 		# v_19 params -------------
	# 		# "cg_1": [
	# 		# 	"Epsilon_3_6.2",
	# 		# 	"model_global-[256, 'ln2', True, 1, '']_[0, 3, 0, 2, 'vanilla', '']_['elu', None]_0.0004_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.98__0"
	# 		# ],
	# 		# "cg_5": [
	# 		# 	"Epsilon_3_6.1",
	# 		# 	"model_global-[128, 'ln2', True, 1, '']_[0, 3, 0, 2, 'vanilla', '']_['elu', None]_0.0002_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.97__0"
	# 		# ],
	# 		# "cg_10": [
	# 		# 	"Epsilon_3_6",
	# 		# 	"model_global-[128, 'ln2', True, 1, '']_[0, 3, 0, 2, 'vanilla', '']_['elu', None]_0.0001_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.97__0"
	# 		# ],
	# 	},
	# 	"interface": {
	# 		# Interface versions with vanilla block.
	# 		# v_23 params -------------
	# 		# "cg_1": [
	# 		# 	"Epsilon_3_13",
	# 		# 	"model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0002_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.97__0"
				
	# 		# ],
	# 		# "cg_5": [
	# 		# 	"Epsilon_3_13.1",
	# 		# 	"model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0002_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.93__0"
	# 		# ],
	# 		# "cg_10": [
	# 		# 	"Epsilon_3_13.2",
	# 		# 	"model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0002_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.93__0"
	# 		# ]
	# 		# v_21 params -------------
	# 		"cg_1": [
	# 			"Epsilon_3_11.1",
	# 			"model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0002_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.95__0"
				
	# 		],
	# 		"cg_5": [
	# 			"Epsilon_3_11.3",
	# 			"model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0001_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.92__0"
	# 		],
	# 		"cg_10": [
	# 			"Epsilon_3_11.4",
	# 			"model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0001_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.89__0"
	# 		]
	# 		# v_19 params -------------
	# 		# "cg_1": [
	# 		# 	"Epsilon_3_5.2",
	# 		# 	"model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0002_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.98__0"
	# 		# ],
	# 		# "cg_5": [
	# 		# 	"Epsilon_3_5.1",
	# 		# 	"model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0001_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.94__0"
	# 		# ],
	# 		# "cg_10": [
	# 		# 	"Epsilon_3_5",
	# 		# 	"model_global-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0001_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.93__0"
	# 		# ]
	# 	}
	# }

	# return params

