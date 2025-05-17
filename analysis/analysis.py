import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import json
import random
import os

import torch
from torch import nn

from dataset.utility import ranges

from src.metrics import torch_metrics
from src.utils import plot_reliabity_diagram

MAX_LEN_DICT = {19: 100, 21: 200, 23: 250}

class JudgementDay():
	def __init__( self ):
		# Dataset version.
		self.data_version = 21
		self.model_version = 21
		# path for the dataset dir.
		self.base_path = f"../database/"
		# Seed for the PRNGs.
		self.global_seed = 11
		# Cutoff to dfine contact for disobind predictions.
		self.contact_threshold = 0.5
		# ipTM cutoff for confident predictions.
		self.iptm_cutoff = 0.0
		# Max prot1/2 lengths.
		self.max_len = MAX_LEN_DICT[self.model_version]
		self.pad = True
		self.mode = "ood" # "ood" or "peds" or "misc"

		self.device = "cuda"

		if self.mode == "ood":
			self.base_af_dir = f"./AF_preds_v{self.data_version}/"
			self.diso_base_dir = f"./Predictions_{self.mode}_v_{self.data_version}/"
			# OOD set target contact maps file.
			self.target_cmap = f"{self.base_path}v_{self.data_version}/Target_bcmap_test_v_{self.data_version}.h5"
			# Disobind predictions file.
			self.disobind_preds = f"{self.diso_base_dir}Disobind_Predictions.npy"

		elif self.mode == "peds":
			self.base_af_dir = f"./AF_peds_preds_v{self.data_version}/"
			self.diso_base_dir = f"./Predictions_{self.mode}_v_{self.data_version}/"
			# OOD set target contact maps file.
			self.target_cmap = f"{self.base_path}PEDS/ped_test_target.h5"
			# Disobind predictions file.
			self.disobind_preds = f"{self.diso_base_dir}Disobind_Predictions_peds.npy"

		elif self.mode == "misc":
			self.base_af_dir = f"./AF_misc_preds_v{self.data_version}/"
			self.diso_base_dir = f"./Predictions_{self.mode}_v_{self.data_version}/"
			# OOD set target contact maps file.
			self.target_cmap = f"{self.base_path}Misc/misc_test_target.h5"
			# Disobind predictions file.
			self.disobind_preds = f"{self.diso_base_dir}Disobind_Predictions_misc.npy"

		else:
			raise ValueError( "Incorrect mode specified (ood/peds/misc supported)..." )

		# AF2 predictions file.
		self.af2m_preds = f"{self.base_af_dir}Predictions_af2m_results_{self.iptm_cutoff}.npy"
		# AF3 predictions file.
		self.af3_preds = f"{self.base_af_dir}Predictions_af3_results_{self.iptm_cutoff}.npy"

		# Predictions from AIUPred and DeepDisoBind
		self.other_methods = "./other_methods/other_methods.npy"

		# # OOD set target contact maps file.
		# self.target_cmap = f"{self.diso_base_dir}/Target_bcmap_test_v_{self.data_version}.h5"
		# self.target_cmap = f"{self.base_path}v_{self.data_version}/Target_bcmap_test_v_{self.data_version}.h5"
		# Fraction of positives in the dataset for all tasks.
		self.fraction_positives = f"{self.base_path}v_{self.data_version}/T5/global-None/fraction_positives.json"
		# File containing info about the merged binary complexes in the dataset.
		self.merged_binary_complexes_dir = f"{self.base_path}v_{self.data_version}/merged_binary_complexes/"

		self.output_dir = f"./Analysis_{self.mode.upper()}_{self.data_version}_{self.iptm_cutoff}/"

		# # PEDS entries.
		# self.target_peds = f"{self.base_path}PEDS/ped_test_target.h5"
		# self.peds_preds = f"./Predictions_peds_v_{self.data_version}/Disobind_Predictions_peds.npy"

		# File to store all results.
		self.full_results_file = f"{self.output_dir}Results_OOD_set_{self.data_version}.csv"
		self.subset_results_file = f"{self.output_dir}Results_OOD_set_subset_{self.data_version}.csv"
		self.other_methods_result_file = f"{self.output_dir}Results_other_methods_{self.data_version}.csv"
		# self.peds_results_file = f"{self.output_dir}Results_PEDS.csv"
		
		# Files for the plots and raw data.
		self.af_conf_pred_counts_file = f"{self.output_dir}Confident_AF_preds_{self.data_version}.txt"
		self.af_confidence_file = f"{self.output_dir}AF_confidence_plot_{self.data_version}"
		self.af_confidence_scores = f"{self.output_dir}AF_confidence_scores_{self.data_version}.csv"
		self.sparsity_file = f"{self.output_dir}Sparsity_F1_plot_{self.data_version}"
		self.case_specific_analysis_file = f"{self.output_dir}Case_sp_analysis_{self.data_version}"
		self.top_preds_diso_af2_file = f"{self.output_dir}Top_preds_Diso_AF2_{self.data_version}"
		self.top_preds_file = f"{self.output_dir}Top_preds_{self.data_version}"


	def seed_worker( self ):
		# Set the seeds for PRNG.
		torch.manual_seed( self.global_seed )
		torch.cuda.manual_seed_all( self.global_seed )
		np.random.seed( self.global_seed )
		random.seed( self.global_seed )


	def forward( self ):
		if not os.path.exists( self.output_dir ):
			os.makedirs( self.output_dir )
		self.seed_worker()

		# Load all predictions.
		self.af2m_preds = np.load( self.af2m_preds, allow_pickle = True ).item()
		self.af3_preds = np.load( self.af3_preds, allow_pickle = True ).item()
		self.disobind_preds = np.load( self.disobind_preds, allow_pickle = True ).item()
		self.target_cmap = h5py.File( self.target_cmap, "r" )
		if self.mode not in ["peds", "misc"]:
			other_methods = np.load( self.other_methods, allow_pickle = True ).item()
			self.aiupred = other_methods["aiupred"]
			self.deepdisobind = other_methods["deepdisobind"]
			self.morfchibi = other_methods["morfchibi"]
		# self.peds_preds = np.load( self.peds_preds, allow_pickle = True ).item()
		# self.target_peds = h5py.File( self.target_peds, "r" )
		self.fraction_positives = json.load( open( self.fraction_positives, "r" ) )

		self.count_confident_AF_predictions()
		self.eval_performance()
		# self.eval_peds_performance()
		# self.eval_single()


	def get_tasks( self ):
		tasks = []
		for obj in ["interaction", "interface"]:
			for cg in [1, 5, 10]:
				tasks.append( f"{obj}_{cg}" )

		return tasks


	def prepare_target( self, target, task ):
		if self.pad:
			max_len = [self.max_len, self.max_len]
		else:
			max_len = [target.shape]
		
		mask = np.zeros( max_len )
		h, w = target.shape
		mask[:h,:w] = target
		target = mask

		target = torch.from_numpy( target )

		_, cg = task.split( "_" )
		cg = int( cg )
		if cg > 1:
			# MaxPool2d requires input of shape ( N, C, L1, L2 ).
			# 	target: ( N, L1, L2 ) --> ( N, 1, L1, L2 )
			eff_len = [max_len[0]//cg, max_len[1]//cg]
			
			with torch.no_grad():
				target = target.unsqueeze( 0 )
				m = nn.MaxPool2d( kernel_size = cg, stride = cg )
				# print( cg, "\t", target.size() )
				target = m( target )
				# Remove the extra C dimension.
				target = target.squeeze( 0 )
		else:
			eff_len = max_len

		if "interface" in task:
			# Find indexes of conatct elements.
			idx = torch.where( target == 1 )
			# Get interacting residues for prot1.
			p1_target = torch.zeros( ( target.shape[0], 1 ) )
			p1_target[idx[0], :] = 1

			# Get interacting residues for prot2.
			p2_target = torch.zeros( ( target.shape[1], 1 ) )
			p2_target[idx[1], :] = 1
			
			# target: ( N, L1, L2 ) --> ( N, L1+L2 )
			target = torch.cat( ( p1_target, p2_target ), axis = 0 )

		return target


	def get_random_baseline( self, target, task ):
		# Fraction of contacts in train set.
		obj, cg = task.split( "_" )
		p = self.fraction_positives[obj][cg]
		n = 1 - p

		random_preds = np.random.choice( [0, 1], ( target.shape ), [n, p] )

		return random_preds


	def create_ood_set_tensors( self ):
		"""
		Generator function that loads all predictions, masks, target cmap, disorder matrices 
			from Disobind, AF2, AF3 OOD prediction files, create a batch for all OOD entries 
			and yields an OOD dict for each task.

		Input:
		----------
		Does not take any arguments.

		Yields:
		----------
		task --> (str) interaction/interface prediction at specific coarse-grained resolution.
		ood_dict --> (dict) containing Disobind, AF2, AF3 predictions, target masks, disorder matrices, 
					and target cmaps for all OOD entries.
		"""
		# For all tasks.		for task in self.get_tasks():
		
		for task in self.get_tasks():
			counts = [0, 0, 0]
			ood_keys = ["AF2_pLDDT_PAE", "AF3_pLDDT_PAE",
					"Disobind_uncal", "Random_baseline", "masks", 
					"disorder_mat1", "disorder_mat2", "order_mat",
					"targets"]
			if self.mode not in ["peds", "misc"]:
				ood_keys.extend( ["Aiupred", "Deepdisobind", "Morfchibi"] )

			# All fields to be included for the results.
			ood_dict = {key:[] for key in ood_keys}
			
			# For all entries.
			entry_ids = []
			for idx, key1 in enumerate( self.target_cmap.keys() ):
				target = np.array( self.target_cmap[key1] )
				target = self.prepare_target( target, task )

				# Ignoring this entry, as AF2-multimer crashed for this  (v_19 dataset).
				# if key1 == "P0DTD1:1743:1808--P0DTD1:1565:1641_1":
					# continue

				u1, u2 = key1.split( "--" )
				u1 = u1.split( ":" )[0]
				u2, c = u2.split( "_" )
				u2 = u2.split( ":" )[0]

				# # "2mps"
				# if f"{u1}--{u2}_{c}" not in ["Q00987--O15350_0"]:
				# 	continue
				# # "2n3a"
				# if f"{u1}--{u2}_{c}" not in ["Q7Z3K3--O75475_0"]:
				# 	continue
				# # "2mkr"
				# if f"{u1}--{u2}_{c}" not in ["P32776--P12978_0"]:
				# 	continue
				# # "1hui"
				# if f"{u1}--{u2}_{c}" not in ["P01308--P01308_0"]:
				# 	continue
				# # "5tmx"
				# if f"{u1}--{u2}_{c}" not in ["P23308--P23308_0"]:
				# 	continue
				# # "2n0p"
				# if f"{u1}--{u2}_{c}" not in ["Q9BV68--P46379_0"]:
				# 	continue
				# # "2dt7"
				# if f"{u1}--{u2}_{c}" not in ["Q12874--Q15459_0"]:
				# 	continue
				# # 2jwn
				# if f"{u1}--{u2}_{c}" not in ["Q6TY21--Q6TY21_0"]:
				# 	continue

				# These Uniprot pairs are sequence redundant with PDB70 at 20% seq identity.
				# 	Ignoring these from evaluation (v_19 dataset).
				# if f"{u1}--{u2}_{c}" in ["P0DTC9--P0DTD1_2", "Q96PU5--Q96PU5_0", "P0AG11--P0AG11_4", 
				# 						"Q9IK92--Q9IK91_0", "Q16236--O15525_0", "P12023--P12023_0",
				# 						"O85041--O85043_0", "P25024--P10145_0"]:
				# 	continue

				entry_ids.append( key1 )

				# For all fields in the prediction dict.
				for key2 in ood_dict.keys():
					if "AF2" in key2:
						ood_dict[key2].append( self.af2m_preds[key1][task][key2] )
					
					elif "AF3" in key2:
						ood_dict[key2].append( self.af3_preds[key1][task][key2] )

					elif "Disobind" in key2:
						ood_dict[key2].append( self.disobind_preds[key1][task][key2] )

					elif "Aiupred" in key2 and self.mode not in ["peds", "misc"]:
						if task == "interface_1":
							ood_dict[key2].append( self.aiupred[f"{u1}--{u2}_{c}"] )
						# Empty arrays for all other tasks.
						else:
							dummy = self.disobind_preds[key1][task]["Disobind_uncal"]
							ood_dict[key2].append( np.zeros( dummy.shape ) )

					elif "Deepdisobind" in key2 and self.mode not in ["peds", "misc"]:
						if task == "interface_1":
							ood_dict[key2].append( self.deepdisobind[f"{u1}--{u2}_{c}"] )
						# Empty arrays for all other tasks.
						else:
							dummy = self.disobind_preds[key1][task]["Disobind_uncal"]
							ood_dict[key2].append( np.zeros( dummy.shape ) )

					elif "Morfchibi" in key2 and self.mode not in ["peds", "misc"]:
						if task == "interface_1":
							ood_dict[key2].append( self.morfchibi[f"{u1}--{u2}_{c}"] )
						# Empty arrays for all other tasks.
						else:
							dummy = self.disobind_preds[key1][task]["Disobind_uncal"]
							ood_dict[key2].append( np.zeros( dummy.shape ) )

					elif key2 in ["masks", "disorder_mat1", "disorder_mat2", "order_mat"]:
						ood_dict[key2].append( self.disobind_preds[key1][task][key2] )

					elif key2 == "targets":
						ood_dict[key2].append( target*self.disobind_preds[key1][task]["masks"] )

					elif "Random" in key2:
						random_preds = self.get_random_baseline( target, task )
						random_preds = random_preds*self.disobind_preds[key1][task]["masks"]

						ood_dict[key2].append( random_preds )

			for key in ood_dict.keys():
				ood_dict[key] = np.stack( ood_dict[key] )

			yield task, ood_dict, entry_ids



	def eval_performance( self ):
		"""
		Calculate metrics for OOD set across all tasks and models 
			and save the results on disk.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		results_dict = {key:[] for key in ["Objective", "CG", "Model", "Recall", "Precision", "F1-score"]}

		for task, ood_dict, entry_ids in self.create_ood_set_tensors():
			print( f"Task {task}..." )
			preds_dict = {}

			# Obtain all raw predictions and combinations of predictions to be tested.
			for key in ood_dict.keys():
				if key not in ["masks", "disorder_mat1", "disorder_mat2", "targets"]:
					preds_dict[key] = torch.from_numpy( ood_dict[key] )

			# AF2/AF3 pLDDT+PAE corrected predictions combined with Disobind predictions.
			b, m, n = ood_dict["AF2_pLDDT_PAE"].shape
			af2_diso = np.stack( [ood_dict["AF2_pLDDT_PAE"].reshape( b, m*n ), ood_dict["Disobind_uncal"].reshape( b, m*n )], axis = 1 )
			af2_diso = np.max( af2_diso, axis = 1 ).reshape( b, m, n )
			preds_dict["AF2_Disobind_uncal"] = torch.from_numpy( af2_diso )

			af3_diso = np.stack( [ood_dict["AF3_pLDDT_PAE"].reshape( b, m*n ), ood_dict["Disobind_uncal"].reshape( b, m*n )], axis = 1 )
			af3_diso = np.max( af3_diso, axis = 1 ).reshape( b, m, n )
			preds_dict["AF3_Disobind_uncal"] = torch.from_numpy( af3_diso )

			# AF2/AF3 pLDDT+PAE corrected predictions for IDR-IDR and IDR-any interactions.
			preds_dict["AF2_IDR-IDR"] = torch.from_numpy( ood_dict["AF2_pLDDT_PAE"]*ood_dict["disorder_mat1"] )
			preds_dict["AF3_IDR-IDR"] = torch.from_numpy( ood_dict["AF3_pLDDT_PAE"]*ood_dict["disorder_mat1"] )
			# Disobind predictions for IDR-IDR and IDR-any interactions.
			preds_dict["Disobind_uncal_IDR-IDR"] = torch.from_numpy( ood_dict["Disobind_uncal"]*ood_dict["disorder_mat1"] )
			preds_dict["AF2_Disobind_uncal_IDR-IDR"] = torch.from_numpy( af2_diso*ood_dict["disorder_mat1"] )

			# Ordered residue interactions.
			preds_dict["AF2_order"] = torch.from_numpy( ood_dict["AF2_pLDDT_PAE"]*ood_dict["order_mat"] )
			preds_dict["AF3_order"] = torch.from_numpy( ood_dict["AF3_pLDDT_PAE"]*ood_dict["order_mat"] )
			# Disobind predictions for ordered residue interactions.
			preds_dict["Disobind_uncal_order"] = torch.from_numpy( ood_dict["Disobind_uncal"]*ood_dict["order_mat"] )
			preds_dict["AF2_Disobind_uncal_order"] = torch.from_numpy( af2_diso*ood_dict["order_mat"] )

			if "interaction" in task:
				preds_dict["AF2_IDR-any"] = torch.from_numpy( ood_dict["AF2_pLDDT_PAE"]*ood_dict["disorder_mat2"] )
				preds_dict["AF3_IDR-any"] = torch.from_numpy( ood_dict["AF3_pLDDT_PAE"]*ood_dict["disorder_mat2"] )

			if "interaction" in task:
				preds_dict["Disobind_uncal_IDR-any"] = torch.from_numpy( ood_dict["Disobind_uncal"]*ood_dict["disorder_mat2"] )
				preds_dict["AF2_Disobind_uncal_IDR-any"] = torch.from_numpy( af2_diso*ood_dict["disorder_mat2"] )

			# now calculate the metrics for all predictions.
			for key in preds_dict.keys():
				obj, cg = task.split( "_" )

				metrics = self.calculate_metrics( pred = preds_dict[key].to( self.device ), 
													target = torch.from_numpy( ood_dict["targets"] ).to( self.device ), 
													multidim_avg = "global" )
				results_dict["Objective"].append( obj.title() )
				results_dict["CG"].append( cg )
				results_dict["Model"].append( key )
				results_dict["Recall"].append( metrics[0] )
				results_dict["Precision"].append( metrics[1] )
				results_dict["F1-score"].append( metrics[2] )

			if task == "interface_1":
				self.eval_other_methods( preds_dict["Disobind_uncal"], preds_dict["Aiupred"],
										preds_dict["Deepdisobind"], preds_dict["Morfchibi"],
										torch.from_numpy( ood_dict["targets"] ) )
			# 	# Create contact density vs interface 1 prediction plots.
			# 	self.create_performance_contactdensity_plots( entry_ids, 
			# 													preds_dict["Disobind_uncal"], 
			# 													preds_dict["AF2_pLDDT_PAE"], 
			# 													ood_dict["targets"] )

			# Do for all tasks.
			# self.create_calibration_plots( preds_dict["Disobind_uncal"], preds_dict["Disobind_cal"], ood_dict["targets"], task )

			# Just adding a empty row for separating different tasks.
			for key in results_dict.keys():
				results_dict[key].append( "" )

		self.create_sparsity_f1_plots( results_dict )
		# self.case_specific_analysis()

		# Dump all calculated metrics on disk.
		df = pd.DataFrame( results_dict )
		df.to_csv( self.full_results_file )

		# Dump a subset of results o disk - for ease of looking.
		subset_dict = {key:[] for key in results_dict.keys() if key not in ["AvgPrecision", "MCC", "AUROC", "Accuracy"]}
		for i in range( len( results_dict["Model"] ) ):
			if results_dict["Model"][i] in ["AF2_pLDDT_PAE", "AF3_pLDDT_PAE","Disobind_uncal",
											"AF2_Disobind_uncal", "AF3_Disobind_uncal",
											"AF2_IDR-IDR", "AF3_IDR-IDR", "Disobind_uncal_IDR-IDR",
											"AF2_Disobind_uncal_IDR-IDR",
											"AF2_order", "AF3_order", "Disobind_uncal_order",
											"AF2_Disobind_uncal_order",
											"AF2_IDR-any", "AF3_IDR-any", "Disobind_uncal_IDR-any",
											"AF2_Disobind_uncal_IDR-any", "Aiupred", "Deepdisobind",
											"Morfchibi", "Random_baseline", ""]:
				for key in subset_dict.keys():
					subset_dict[key].append( results_dict[key][i] )

		df = pd.DataFrame( subset_dict )
		df.to_csv( self.subset_results_file, index = False )



	# def create_performance_contactdensity_plots( self, entries, diso_preds, af2_preds, diso_af2_preds, target ):
	# 	"""
	# 	Create a plot for the performance(Recall/Precision/F1) vs contact density for Interface_1.
	# 	Save the raw data on disk.
	# 	Also save the top predictions (> 0.7 F1 score) on disk.

	# 	Input:
	# 	----------
	# 	entries --> list of all entry_id for all OOD entries.
	# 	diso_preds --> (torch.tensor) Calibrated Disobind prediction.
	# 	af2_preds --> (torch.tensor) AF2 predictions.
	# 	target --> (torch.tensor) binary output labels.

	# 	Returns:
	# 	----------
	# 	None
	# 	"""
	# 	entries = np.array( entries )

	# 	# Get per OOD entry metrics.
	# 	metrics_diso = torch_metrics( preds = diso_preds.to( self.device ), 
	# 							target = torch.from_numpy( target ).to( self.device ), 
	# 							threshold = self.contact_threshold,
	# 							multidim_avg = "samplewise_none", device = self.device )
		

	# 	metrics_af2 = torch_metrics( preds = af2_preds.to( self.device ), 
	# 							target = torch.from_numpy( target ).to( self.device ), 
	# 							threshold = self.contact_threshold,
	# 							multidim_avg = "samplewise_none", device = self.device )

	# 	metrics_diso_af2 = torch_metrics( preds = diso_af2_preds.to( self.device ), 
	# 								target = torch.from_numpy( target ).to( self.device ), 
	# 								threshold = self.contact_threshold,
	# 								multidim_avg = "samplewise_none", device = self.device )
		
	# 	contact_density = []
	# 	for i, id_ in enumerate( entries ):
	# 		head1, head2 = id_.split( "--" )
	# 		head2, num = head2.split( "_" )
	# 		uni_id1, r11, r12 = head1.split( ":" )
	# 		uni_id2, r21, r22 = head2.split( ":" )
	# 		len1 = int( r12 ) - int( r11 ) + 1
	# 		len2 = int( r22 ) - int( r21 ) + 1
			
	# 		contacts = np.count_nonzero( target[i] )
	# 		num_elements = len1 + len2

	# 		contact_density.append( contacts/num_elements )

	# 	contact_density = np.array( contact_density ).reshape( len( entries ), 1 )

	# 	re_diso = metrics_diso[0].cpu().numpy()
	# 	prec_diso = metrics_diso[1].cpu().numpy()
	# 	f1_diso = metrics_diso[2].cpu().numpy()

	# 	re_af2 = metrics_af2[0].cpu().numpy()
	# 	prec_af2 = metrics_af2[1].cpu().numpy()
	# 	f1_af2 = metrics_af2[2].cpu().numpy()

	# 	re_diso_af2 = metrics_diso_af2[0].cpu().numpy()
	# 	prec_diso_af2 = metrics_diso_af2[1].cpu().numpy()
	# 	f1_diso_af2 = metrics_diso_af2[2].cpu().numpy()

	# 	fig, ax = plt.subplots( 1, 3, figsize = ( 20, 8 ) )
	# 	ax[0].scatter( contact_density, re_diso )
	# 	ax[0].set_ylabel( "Recall" )
	# 	ax[0].set_xlabel( "Contact density" )

	# 	ax[1].scatter( contact_density, prec_diso )
	# 	ax[1].set_ylabel( "Precision" )
	# 	ax[1].set_xlabel( "Contact density" )
		
	# 	ax[2].scatter( contact_density, f1_diso )
	# 	ax[2].set_ylabel( "F1 score" )
	# 	ax[2].set_xlabel( "Contact density" )

	# 	plt.savefig( f"{self.contact_density_file}.png", dpi = 300 )
	# 	plt.close()

	# 	# Save samplewsie metrics for Disobind and AF2 on OOD set.
	# 	df = pd.DataFrame()
	# 	df["Entry ID"] = entries
	# 	df["Contact density"] = np.round( contact_density.reshape( -1 ), 3 )
	# 	df["Recall"] = np.round( list( re_diso ), 3 )
	# 	df["Precision"] = np.round( list( prec_diso ), 3 )
	# 	df["F1"] = np.round( list( f1_diso ), 3 )
	# 	df["AF2-Recall"] = np.round( list( re_af2 ), 3 )
	# 	df["AF2-Precision"] = np.round( list( prec_af2 ), 3 )
	# 	df["AF2-F1"] = np.round( list( f1_af2 ), 3 )
	# 	df["Diso+AF2-Recall"] = np.round( list( re_diso_af2 ), 3 )
	# 	df["Diso+AF2-Precision"] = np.round( list( prec_diso_af2 ), 3 )
	# 	df["Diso+AF2-F1"] = np.round( list( f1_diso_af2 ), 3 )
	# 	df.to_csv( f"{self.contact_density_file}.csv" )

	# 	# Save samplewsie metrics for Disobind and AF2 on best performing pairs.
	# 	idx = np.where( f1_diso_af2 > 0.7 )
	# 	top = entries[idx]

	# 	df = pd.DataFrame()
	# 	df["Entry ID"] = top
	# 	df["Contact density"] = np.round( contact_density[idx], 3 )
	# 	df["Recall"] = np.round( re_diso[idx], 3 )
	# 	df["Precision"] = np.round( prec_diso[idx], 3 )
	# 	df["F1 score"] = np.round( f1_diso[idx], 3 )
	# 	df["AF2-Recall"] = np.round( re_af2[idx], 3 )
	# 	df["AF2-Precision"] = np.round( prec_af2[idx], 3 )
	# 	df["AF2-F1 score"] = np.round( f1_af2[idx], 3 )
	# 	df["Diso+AF2-Recall"] = np.round( list( re_diso_af2[idx] ), 3 )
	# 	df["Diso+AF2-Precision"] = np.round( list( prec_diso_af2[idx] ), 3 )
	# 	df["Diso+AF2-F1"] = np.round( list( f1_diso_af2[idx] ), 3 )
	# 	df.to_csv( f"{self.top_preds_file}.csv" )

	# 	plt.scatter( f1_diso[idx], f1_af2[idx], color = "blue" )
	# 	plt.xlim( 0.5, 1 )
	# 	plt.ylim( -0.1, 1 )
	# 	plt.savefig( f"{self.top_preds_diso_af2_file}.png", dpi = 300 )
	# 	plt.close()

	def eval_other_methods( self, diso_preds: torch.tensor, aiupred_preds: torch.tensor,
							deepdisobind_preds: torch.tensor, morfchibi_preds: torch.tensor,
							target: torch.tensor ):
		"""
		Compare Disobind performance with AIUPred, DeepDisoBind, and MORfchibi.
		The later 3 are partner-independent interface predictors for IDRs and
			so we evaluate performance only for the prot1 in all OOD entries
			which is an IDR.
		"""
		# Need to slice out the protein1 interface only from diso_preds and target.
		diso_p1 = diso_preds[:,:self.max_len].to( self.device )
		aiupred_p1 = aiupred_preds[:,:self.max_len].to( self.device )
		deepdisobind_p1 = deepdisobind_preds[:,:self.max_len].to( self.device )
		morfchibi_p1 = morfchibi_preds[:,:self.max_len].to( self.device )
		target_p1 = target[:,:self.max_len].to( self.device )

		diso_metrics = self.calculate_metrics( pred = diso_p1,
													target = target_p1,
													multidim_avg = "global" )
		aiupred_metrics = self.calculate_metrics( pred = aiupred_p1,
													target = target_p1,
													multidim_avg = "global" )
		deepdisobind_metrics = self.calculate_metrics( pred = deepdisobind_p1,
														target = target_p1,
														multidim_avg = "global" )
		morfchibi_metrics = self.calculate_metrics( pred = morfchibi_p1,
													target = target_p1,
													multidim_avg = "global" )


		df = pd.DataFrame()
		df["Model"] = ["Disobind_uncal", "Aiupred", "Deepdisobind", "Morfchibi"]
		df["Recall"] = [diso_metrics[0], aiupred_metrics[0], deepdisobind_metrics[0], morfchibi_metrics[0]]
		df["Precision"] = [diso_metrics[1], aiupred_metrics[1], deepdisobind_metrics[1], morfchibi_metrics[1]]
		df["F1-score"] = [diso_metrics[2], aiupred_metrics[2], deepdisobind_metrics[2], morfchibi_metrics[2]]
		df.to_csv( self.other_methods_result_file, index = False )



	def eval_single( self ):
		"""
		Calculate metric for performance for a specific model.
		"""
		print( "Evaluation for PEDS..." )
		results_dict = {key:[] for key in ["Objective", "CG", "Recall",
												"Precision", "F1-score"]}
		for task in self.get_tasks():
			print( f"Task {task}..." )
			preds, targets = [], []

			for entry_id in self.target_cmap:
				# # Ignoring this entry, as AF2-multimer crashed for this.
				# if entry_id == "P0DTD1:1743:1808--P0DTD1:1565:1641_1":
				# 	continue

				u1, u2 = entry_id.split( "--" )
				u1 = u1.split( ":" )[0]
				u2, c = u2.split( "_" )
				u2 = u2.split( ":" )[0]

				# # These Uniprot pairs are sequence redundant with PDB70 at 20% seq identity.
				# # 	Ignoring these from evaluation (v_19 dataset).
				# if f"{u1}--{u2}_{c}" in ["P0DTC9--P0DTD1_2", "Q96PU5--Q96PU5_0", "P0AG11--P0AG11_4", 
				# 						"Q9IK92--Q9IK91_0", "Q16236--O15525_0", "P12023--P12023_0",
				# 						"O85041--O85043_0", "P25024--P10145_0"]:
				# 	continue

				# # v_19 OOD not in v_221 and v_23 train.
				# if entry_id not in ['Q9UIF9:597:650--Q9UIF9:544:596_0', 'P84092:144:219--Q0JRZ9:318:341_0', 'P07101:128:184--P07101:128:184_6',
				# 			'Q9NPI8:162:215--Q00597:119:177_0', 'P43405:71:134--P43405:204:269_9', 'P25685:199:221--Q9Y266:102:128_0',
				# 			'Q96RI1:258:314--Q96RI1:258:314_0', 'Q9Y618:2334:2354--P37231:299:367_0', 'P07101:109:160--P07101:109:160_1',
				# 			'Q8WUM0:1089:1156--Q8WUM0:546:613_1', 'P53041:85:151--P08238:640:692_0', 'P0AFD6:1:90--P33602:680:735_4',
				# 			'P49789:61:147--P49789:55:108_1', 'Q99ZW2:1034:1088--P68398:77:160_0', 'P0AG30:315:366--P0AG30:315:366_14',
				# 			'P84092:68:135--P63010:513:584_21', 'Q16236:506:559--O15525:23:122_0', 'Q8AVI7:3:74--Q6GP41:2:79_0',
				# 			'B7UM94:22:106--B7UM94:22:106_4', 'P04273:95:193--P04273:95:193_0', 'P37840:69:97--P37840:38:67_9',
				# 			'P03317:2007:2069--P03317:2431:2505_4', 'Q02199:371:470--P48837:502:540_4', 'A0A5F9CI80:2:74--G1STG2:65:139_0',
				# 			'Q07666:143:201--Q07666:143:201_0', 'P01861:118:146--P55899:93:158_1', 'Q92800:80:164--O75530:77:167_0']:
				# 	continue

				# # v_21 OOD not in v_23 train.
				# if entry_id not in ['P37840:69:97--P37840:38:67_5', 'Q96EP0:966:1070--Q96EP0:867:959_3', 'P49366:8:185--P49366:10:78_0',
				# 				'P25685:199:221--Q9Y266:102:128_0', 'P51946:1:38--P51948:244:308_0', 'P84092:144:219--Q0JRZ9:318:341_0',
				# 				'B7UM94:19:192--B7UM94:19:192_0', 'Q92800:80:164--O75530:77:258_0', 'P55011:216:250--P55011:1022:1212_2',
				# 				'P03317:2007:2069--P03317:2358:2505_2', 'P04273:95:193--P04273:95:193_0', 'P37231:231:367--Q9Y618:2334:2354_0',
				# 				'P28307:41:151--P28307:41:151_0', 'Q9HBM1:80:224--Q8NBT2:86:197_0', 'Q16543:102:137--A0A7E5VSK5:267:371_1',
				# 				'P56211:86:112--P63151:8:60_0', 'P06179:1:39--Q9R016:773:898_2', 'A0A6H1PJZ3:263:441--A0A6H1PJZ3:854:1000_4']:
				# 	continue

				# # "2n3a"
				# if f"{u1}--{u2}_{c}" not in ["Q7Z3K3--O75475_0"]:
				# 	continue
				# # "2dt7"
				# if f"{u1}--{u2}_{c}" not in ["Q12874--Q15459_0"]:
				# 	continue
				# # 2jwn
				# if f"{u1}--{u2}_{c}" not in ["Q6TY21--Q6TY21_0"]:
				# 	continue
				# # 2mkr
				# if f"{u1}--{u2}_{c}" not in ["P32776--P12978_0"]:
				# 	continue

				# # "2lmq"
				# if f"{u1}--{u2}_{c}" not in ["P05067--P05067_0"]:
				# 	continue
				# # "8cmk"
				# if f"{u1}--{u2}_{c}" not in ["Q9Y5L0--Q14011_0"]:
				# 	continue
				# # 2mwy
				# if f"{u1}--{u2}_{c}" not in ["O15151--P04637_0"]:
				# 	continue
				# # 2kqs
				# if f"{u1}--{u2}_{c}" not in ["P63165--Q9UER7_0"]:
				# 	continue
				# 5xv8
				if f"{u1}--{u2}_{c}" not in ["Q2YD98--P32780_0"]:
					continue

				# # 7lna
				# if f"{u1}--{u2}_{c}" not in ["P04273--P04273_0"]:
				# 	continue
				# # 6xmn
				# if f"{u1}--{u2}_{c}" not in ["P10145--P25024_0"]:
				# 	continue
				target = self.target_cmap[entry_id]
				target = np.array( self.target_cmap[entry_id] )
				target = self.prepare_target( target, task )

				# PEDS predictions have the same organization as the OOD set predictions.
				preds.append( np.array( self.disobind_preds[entry_id][task]["Disobind_uncal"] ) )
				targets.append( target )

			preds = torch.from_numpy( np.stack( preds ) )
			targets = torch.from_numpy( np.stack( targets ) )

			obj, cg = task.split( "_" )

			metrics = self.calculate_metrics( pred = preds.to( self.device ),
												target = targets.to( self.device ),
												multidim_avg = "global" )

			results_dict["Objective"].append( obj.title() )
			results_dict["CG"].append( cg )
			results_dict["Recall"].append( metrics[0] )
			results_dict["Precision"].append( metrics[1] )
			results_dict["F1-score"].append( metrics[2] )

		df = pd.DataFrame( results_dict )
		df.to_csv( f"{self.output_dir}Results_Disobind_only.csv", index = False )


	def count_confident_AF_predictions( self ):
		"""
		Count the no. of confident AF2 and AF3 predictions based on:
			iptm score for AF2 and AF3.
		Plot AF2-ipTM vs AF3-ipTM.
		Save the raw data on disk.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		counts1, counts2, counts3, counts4, counts5, counts6 = 0, 0, 0, 0, 0, 0
		af2_score, af3_score = [], []

		selected_entries = []
		for key in self.af2m_preds.keys():
			if key == "P0DTD1:1743:1808--P0DTD1:1565:1641_1":
				continue

			u1, u2 = key.split( "--" )
			u1 = u1.split( ":" )[0]
			u2, c = u2.split( "_" )
			u2 = u2.split( ":" )[0]

			# These Uniprot pairs are sequence redundant with PDB70 at 20% seq identity.
			# 	Ignoring these from evaluation.
			if f"{u1}--{u2}_{c}" in ["P0DTC9--P0DTD1_2", "Q96PU5--Q96PU5_0", "P0AG11--P0AG11_4", 
									"Q9IK92--Q9IK91_0", "Q16236--O15525_0", "P12023--P12023_0", "O85041--O85043_0", "P25024--P10145_0"]:
				continue

			selected_entries.append( key )
			af2_score.append( self.af2m_preds[key]["scores"][0] )
			af3_score.append( self.af3_preds[key]["scores"][0] )

			if af2_score[-1] <= self.iptm_cutoff:
				counts1 += 1
			if af3_score[-1] <= self.iptm_cutoff:
				counts2 += 1
			if af2_score[-1] >= 0.8:
				counts3 += 1
			if af3_score[-1] >= 0.8:
				counts4 += 1
			if af2_score[-1] >= self.iptm_cutoff and af3_score[-1] <= self.iptm_cutoff:
				counts5 += 1
			if af2_score[-1] >= af3_score[-1]:
				counts6 += 1

		# print( counts1, "  ", counts2, "  ",counts3, "  ", counts4, "  ", counts5, "  ", counts6 )
		# exit()
		plt.plot( [0, 1], [0, 1], color = "gray" )
		plt.scatter( af2_score, af3_score )
		plt.axvline( self.iptm_cutoff, color = "red" )
		plt.axhline( self.iptm_cutoff, color = "red" )
		plt.xlabel( "AF2 iptm" )
		plt.ylabel( "AF3 iptm" )
		plt.savefig( f"{self.af_confidence_file}.png" , dpi = 300 )
		plt.close()

		df = pd.DataFrame()
		df["Entry_id"] = selected_entries
		df["AF2_iptm"] = af2_score
		df["AF3_iptm"] = af3_score
		df.to_csv( f"{self.af_confidence_file}.csv" )
		
		with open( self.af_conf_pred_counts_file, "w" ) as w:
			w.writelines( f"AF2 preds with ipTM <= {self.iptm_cutoff} = {counts1}\n" )
			w.writelines( f"AF3 preds with ipTM <= {self.iptm_cutoff} = {counts2}\n" )
			w.writelines( f"AF2 preds with ipTM >= 0.8 = {counts3}\n" )
			w.writelines( f"AF3 preds with ipTM >= 0.8 = {counts4}\n" )
			w.writelines( f"AF2  ipTM >= {self.iptm_cutoff} and AF3  ipTM <= {self.iptm_cutoff} = {counts5}\n" )
			w.writelines( f"AF2  ipTM >= AF3  ipTM = {counts6}\n" )



	def create_calibration_plots( self, uncal_pred, cal_pred, target, task ):
		"""
		Create the calibration plots for the uncalibrated and calibrated Disobind preds.
		Save the raw data on disk.

		Input:
		----------
		uncal_pred --> (torch.tensor) Uncalibrated Disobind prediction.
		cal_pred --> (torch.tensor) Calibrated Disobind prediction.
		target --> (torch.tensor) binary output labels.
		task --> (str) identifier for the task (interaction/interface) across all CG (1/5/10).

		Returns:
		----------
		None
		"""
		print( "Creating calibration plot..." )

		os.chdir( self.output_dir )
		plot_reliabity_diagram( uncal_preds = uncal_pred.detach().cpu().numpy().flatten(), 
								cal_preds = cal_pred.detach().cpu().numpy().flatten(), 
								target = target, #.detach().cpu().numpy().flatten(), 
								file_name = task )
		os.chdir( "../" )



	def create_sparsity_f1_plots( self, results_dict ):
		"""
		Create the plot for sparsity in dataset vs the model performance.
		Save the raw data on disk.

		Input:
		----------
		results_dict --> dict containing metrics for all models on all the tasks.

		Returns:
		----------
		None
		"""
		print( "\nCreating aparsity vs F1 score plot... plot..." )

		sparsity = []
		print( self.fraction_positives )
		for key1 in self.fraction_positives.keys():
			for key2 in self.fraction_positives[key1].keys():
				sparsity.append( round( 1 - self.fraction_positives[key1][key2], 4 )*100 )

		f1 = []
		for i in range( len( results_dict["Objective"] ) ):
			if results_dict["Model"][i] == "Disobind_uncal":
				f1.append( results_dict["F1-score"][i] )

		plt.plot( sparsity, f1, marker = "o", color = "blue" )
		plt.xlabel( "Sparsity" )
		plt.ylabel( "F1-score" )
		plt.savefig( f"{self.sparsity_file}.png", dpi = 300 )
		plt.close()

		df = pd.DataFrame()
		df["Sparsity"] = sparsity
		df["F1"] = f1
		df.to_csv( f"{self.sparsity_file}.csv" )




	def case_specific_analysis( self ):
		"""
		Analysis of the top performing OOD pairs.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		top_pairs = ["P25024:1:29--P10145:28:93_0", "P04273:95:193--P04273:95:193_0"]
		fig, ax = plt.subplots( 8, 1, figsize = ( 15, 15 ) )

		w = open( f"{self.case_specific_analysis_file}.txt", "w" )
		i = 0
		for entry_id in top_pairs:
			uni_id1, uni_id2 = entry_id.split( "--" )
			uni_id2, cp = uni_id2.split( "_" )
			uni_id1, s1, e1 = uni_id1.split( ":" )
			uni_id2, s2, e2 = uni_id2.split( ":" )

			id_ = f"{uni_id1}--{uni_id2}_{cp}"

			diso_pred = self.disobind_preds[entry_id]["interface_1"]["Disobind_uncal"]
			diso_pred = np.where( diso_pred >= self.contact_threshold, 1, 0 )

			af2_pred = self.af2m_preds[entry_id]["interface_1"]["AF2_pLDDT_PAE"]
			print( af2_pred.shape )

			af2_diso = np.stack( [diso_pred, af2_pred], axis = 1 )
			print( af2_diso.shape )
			af2_diso = np.max( af2_diso, axis = 1 )
			print( af2_diso.shape )

			target = np.array( self.target_cmap[entry_id] )
			target = self.prepare_target( target, "interface_1" )
			print( target.shape )

			# Write predicted interface residues.
			uni_pos1 = np.arange( int( s1 ), int( e1 ) + 1, 1 )
			uni_pos2 = np.arange( int( s2 ), int( e2 ) + 1, 1 )

			idx1 = np.where( af2_diso[:self.max_len] == 1 )[0]
			idx2 = np.where( af2_diso[self.max_len:] == 1 )[0]

			uni_pos1 = ranges( uni_pos1[idx1] )
			uni_pos1 = [f"{e[0]}-{e[1]}" for e in uni_pos1]
			uni_pos2 = ranges( uni_pos2[idx2] )
			uni_pos2 = [f"{e[0]}-{e[1]}" for e in uni_pos2]
			
			w.writelines( f"{entry_id}\n" )
			w.writelines( f"Predicted prot1 interface: {','.join( uni_pos1 )}\n" )
			w.writelines( f"Predicted prot2 interface: {','.join( uni_pos2 )}\n" )

			# Write target interface residues.
			uni_pos1 = np.arange( int( s1 ), int( e1 ) + 1, 1 )
			uni_pos2 = np.arange( int( s2 ), int( e2 ) + 1, 1 )

			idx1 = np.where( target[:self.max_len] == 1 )[0]
			idx2 = np.where( target[self.max_len:] == 1 )[0]

			uni_pos1 = ranges( uni_pos1[idx1] )
			uni_pos1 = [f"{e[0]}-{e[1]}" for e in uni_pos1]
			uni_pos2 = ranges( uni_pos2[idx2] )
			uni_pos2 = [f"{e[0]}-{e[1]}" for e in uni_pos2]

			w.writelines( f"Target prot1 interface: {','.join( uni_pos1 )}\n" )
			w.writelines( f"Target prot2 interface: {','.join( uni_pos2 )}\n" )

			w.writelines( "\n-----------------------------------------------\n" )
		w.close()



	def calculate_metrics( self, pred, target, multidim_avg ):
		"""
		Calculate the following metrics:
			Recall, Precision, F1score, AvgPrecision, MCC, AUROC, Accuracy.

		Input:
		----------
		pred --> (torch.tensor) model output.
		target --> (torch.tensor) binary output labels.
		multidim_avg --> averaging scheme - global/samplewise/samplewise-none.

		Returns:
		----------
		metric_array --> np.array containing the calculated metric values in order:
			Recall, Precision, F1score, AvgPrecision, MCC, AUROC, Accuracy.	
		"""
		metrics = torch_metrics( pred, target, self.contact_threshold, multidim_avg, self.device )
		metric_array = np.array( [
					metrics[0].item(),
					metrics[1].item(),
					metrics[2].item(),
					metrics[3].item(),
					metrics[4].item(),
					metrics[5].item(),
					metrics[6].item()
					] ) #.reshape( 1, 7 )

		return np.round( metric_array, 2 )


if __name__ == "__main__":
	JudgementDay().forward()
	print( "May the Force be with you..." )
