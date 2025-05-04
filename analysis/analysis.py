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


class JudgementDay():
	def __init__( self ):
		# Dataset version.
		self.version = 21
		# path for the dataset dir.
		self.base_path = f"../database/"
		# Seed for the PRNGs.
		self.global_seed = 11
		# Cutoff to dfine contact for disobind predictions.
		self.contact_threshold = 0.5
		# ipTM cutoff for confident predictions.
		self.iptm_cutoff = 0.0
		# Max prot1/2 lengths.
		self.max_len = 200
		self.pad = True
		self.device = "cuda"

		# AF2 predictions file.
		self.af2m_preds = f"./AF_preds_v{self.version}/Predictions_af2m_results_{self.iptm_cutoff}.npy"
		# AF3 predictions file.
		self.af3_preds = f"./AF_preds_v{self.version}/Predictions_af3_results_{self.iptm_cutoff}.npy"
		# Disobind predictions file.
		self.disobind_preds = f"./Predictions_ood_v{self.version}/Disobind_Predictions.npy"

		# OOD set target contact maps file.
		self.target_cmap = f"{self.base_path}v_{self.version}/Target_bcmap_test_v_{self.version}.h5"
		# Fraction of positives in the dataset for all tasks.
		self.fraction_positives = f"{self.base_path}v_{self.version}/T5/global-None/fraction_positives.json"
		# File containing info about the merged binary complexes in the dataset.
		self.merged_binary_complexes_dir = f"{self.base_path}v_{self.version}/merged_binary_complexes/"

		self.output_dir = f"./Analysis_OOD_{self.version}_{self.iptm_cutoff}/"

		# File to store all results.
		self.full_results_file = f"{self.output_dir}Results_OOD_set_{self.version}.csv"
		self.subset_results_file = f"{self.output_dir}Results_OOD_set_subset_{self.version}.csv"
		
		# Files for the plots and raw data.
		self.af_conf_pred_counts_file = f"{self.output_dir}Confident_AF_preds_{self.version}.txt"
		self.af_confidence_file = f"{self.output_dir}AF_confidence_plot_{self.version}"
		self.af_confidence_scores = f"{self.output_dir}AF_confidence_scores_{self.version}.csv"
		self.sparsity_file = f"{self.output_dir}Sparsity_F1_plot_{self.version}"
		self.case_specific_analysis_file = f"{self.output_dir}Case_sp_analysis_{self.version}"
		self.top_preds_diso_af2_file = f"{self.output_dir}Top_preds_Diso_AF2_{self.version}"
		self.top_preds_file = f"{self.output_dir}Top_preds_{self.version}"


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
		self.fraction_positives = json.load( open( self.fraction_positives, "r" ) )

		self.count_confident_AF_predictions()
		self.eval_performance()


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
		# For all tasks.
		for task in self.get_tasks():
			counts = [0, 0, 0]
			# All fields to be included for the results.
			ood_dict = {key:[] for key in ["AF2_pLDDT_PAE", "AF3_pLDDT_PAE",
											"Disobind_uncal", "Random_baseline", "masks", 
											"disorder_mat1", "disorder_mat2", "order_mat",
											"targets"]}

			# For all entries.
			entry_ids = []
			for idx, key1 in enumerate( self.target_cmap.keys() ):
				target = np.array( self.target_cmap[key1] )
				target = self.prepare_target( target, task )

				# Ignoring this entry, as AF2-multimer crashed for this.
				if key1 == "P0DTD1:1743:1808--P0DTD1:1565:1641_1":
					continue

				u1, u2 = key1.split( "--" )
				u1 = u1.split( ":" )[0]
				u2, c = u2.split( "_" )
				u2 = u2.split( ":" )[0]

				# These Uniprot pairs are sequence redundant with PDB70 at 20% seq identity.
				# 	Ignoring these from evaluation.
				if f"{u1}--{u2}_{c}" in ["P0DTC9--P0DTD1_2", "Q96PU5--Q96PU5_0", "P0AG11--P0AG11_4", 
										"Q9IK92--Q9IK91_0", "Q16236--O15525_0", "P12023--P12023_0",
										"O85041--O85043_0", "P25024--P10145_0"]:
					continue

				entry_ids.append( key1 )

				# For all fields in the prediction dict.
				for key2 in ood_dict.keys():
					if "AF2" in key2:
						ood_dict[key2].append( self.af2m_preds[key1][task][key2] )
					
					elif "AF3" in key2:
						ood_dict[key2].append( self.af3_preds[key1][task][key2] )

					elif "Disobind" in key2:
						ood_dict[key2].append( self.disobind_preds[key1][task][key2] )

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

			# if task == "interface_1":
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
											"AF2_Disobind_uncal_IDR-any", ""]:
				for key in subset_dict.keys():
					subset_dict[key].append( results_dict[key][i] )

		df = pd.DataFrame( subset_dict )
		df.to_csv( self.subset_results_file )



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
