####### Store parameter files anme for each kinda model. #######
######## ------>"May the Force serve u well..." <------#########
################################################################

def parameter_files():
	params = {
		"interaction": {
			# Multi output interaction versions with vanilla block.
			"cg_1": [
				"Epsilon_3_0",
				"model_global_-[256, 'ln2', True, 1, '']_[0, 3, 0, 2, 'vanilla', '']_['elu', None]_0.0004_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.98__0"
			],
			"cg_5": [
				"Epsilon_3_0.1",
				"model_global_-[128, 'ln2', True, 1, '']_[0, 3, 0, 2, 'vanilla', '']_['elu', None]_0.0002_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.97__0"
			],
			"cg_10": [
				"Epsilon_3_0.2",
				"model_global_-[128, 'ln2', True, 1, '']_[0, 3, 0, 2, 'vanilla', '']_['elu', None]_0.0001_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.97__0"
			],
		},
		"interface": {
			# Interface versions with vanilla block.
			"cg_1": [
				"Epsilon_3_1",
				"model_global_-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0002_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.98__0"
			],
			"cg_5": [
				"Epsilon_3_1.1",
				"model_global_-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0001_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.94__0"
			],
			"cg_10": [
				"Epsilon_3_1.2",
				"model_global_-[128, 'ln2', True, 1, '']_[0, 0, 0, 0, 'vanilla', '']_['elu', None]_0.0001_0.5_[0.9, 3]_[0.2, 0, 0, 0, 0]_0.05_0.93__0"
			]
		}
	}

	return params

