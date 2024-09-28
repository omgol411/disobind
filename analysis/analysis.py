import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import json
import random
import os

import torch
from torch import nn

from src.metrics import torch_metrics
from src.utils import plot_reliabity_diagram


class JudgementDay():
	def __init__( self ):
		# Dataset version.
		self.version = 19
		# path for the dataset dir.
		self.base_path = f"/data2/kartik/Disorder_Proteins/disobind_archive/Database/"
		# Seed for the PRNGs.
		self.global_seed = 11
		# Cutoff to dfine contact for disobind predictions.
		self.contact_threshold = 0.5
		# ipTM cutoff for confident predictions.
		self.iptm_cutoff = 0.75
		# Max prot1/2 lengths.
		self.max_len = 100
		self.pad = True
		self.device = "cuda"

		# AF2 predictions file.
		self.af2m_preds = f"./AF_preds_v{self.version}/Predictions_af2m_results_{self.iptm_cutoff}.npy"
		# AF3 predictions file.
		self.af3_preds = f"./AF_preds_v{self.version}/Predictions_af3_results_{self.iptm_cutoff}.npy"
		# Disobind predictions file.
		self.disobind_preds = f"./Predictions_ood_v{self.version}/Disobind_Predictions.npy"

		# OOD set target contact maps file.
		self.target_cmap = f"{self.base_path}v_{self.version}/Output_bcmap_test_v_{self.version}.h5"
		# Fraction of positives in the dataset for all tasks.
		self.fraction_positives = f"{self.base_path}v_{self.version}/T5/global-None/fraction_positives.json"
		# File containing info about the merged binary complexes in the dataset.
		self.merged_binary_complexes_dir = f"{self.base_path}v_{self.version}/merged_binary_complexes/"

		self.output_dir = f"./Analysis_OOD_{self.version}_{self.iptm_cutoff}/"

		# File to store all results.
		self.full_results_file = f"{self.output_dir}Results_OOD_set_{self.version}.csv"
		self.subset_results_file = f"{self.output_dir}Results_OOD_set_subset_{self.version}.csv"
		
		# Files for the plots and raw data.
		self.interface_plots_file = f"{self.output_dir}OOD_plots_{self.version}"
		self.af_conf_pred_counts_file = f"{self.output_dir}Confident_AF_preds_{self.version}.txt"
		self.af_confidence_file = f"{self.output_dir}AF_confidence_plot_{self.version}"
		self.af_confidence_scores = f"{self.output_dir}AF_confidence_scores_{self.version}.csv"
		self.sparsity_file = f"{self.output_dir}Sparsity_F1_plot_{self.version}"
		self.contact_density_file = f"{self.output_dir}Contact_density_Metrics_{self.version}"
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
			max_len = [100, 100]
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
											"Disobind_uncal", "Disobind_cal", "Random_baseline", "masks", 
											"disorder_mat1", "disorder_mat2", "targets"]}

			# For all entries.
			entry_ids = []
			for idx, key1 in enumerate( self.target_cmap.keys() ):
				target = np.array( self.target_cmap[key1] )
				target = self.prepare_target( target, task )

				# Ignoring this entry, as AF2-multimer crashed for this.
				if key1 == "P0DTD1:1743:1808--P0DTD1:1565:1641_1":
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

					elif key2 in ["masks", "disorder_mat1", "disorder_mat2"]:
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
				if key not in ["masks", "disorder_mat1", "disorder_mat2", "disorder_promoting_aa", "slims", "targets"]:
					preds_dict[key] = torch.from_numpy( ood_dict[key] )

			# AF2/AF3 pLDDT+PAE corrected predictions combined with Disobind predictions.
			b, m, n = ood_dict["AF2_pLDDT_PAE"].shape
			af2_diso = np.stack( [ood_dict["AF2_pLDDT_PAE"].reshape( b, m*n ), ood_dict["Disobind_uncal"].reshape( b, m*n )], axis = 1 )
			af2_diso = np.max( af2_diso, axis = 1 ).reshape( b, m, n )
			preds_dict["AF2_Disobind_uncal"] = torch.from_numpy( af2_diso )
			
			af2_diso = np.stack( [ood_dict["AF2_pLDDT_PAE"].reshape( b, m*n ), ood_dict["Disobind_cal"].reshape( b, m*n )], axis = 1 )
			af2_diso = np.max( af2_diso, axis = 1 ).reshape( b, m, n )
			preds_dict["AF2_Disobind_cal"] = torch.from_numpy( af2_diso )

			af3_diso = np.stack( [ood_dict["AF3_pLDDT_PAE"].reshape( b, m*n ), ood_dict["Disobind_uncal"].reshape( b, m*n )], axis = 1 )
			af3_diso = np.max( af3_diso, axis = 1 ).reshape( b, m, n )
			preds_dict["AF3_Disobind_uncal"] = torch.from_numpy( af3_diso )

			af3_diso = np.stack( [ood_dict["AF3_pLDDT_PAE"].reshape( b, m*n ), ood_dict["Disobind_cal"].reshape( b, m*n )], axis = 1 )
			af3_diso = np.max( af3_diso, axis = 1 ).reshape( b, m, n )
			preds_dict["AF3_Disobind_cal"] = torch.from_numpy( af3_diso )
			
			# AF2/AF3 pLDDT+PAE corrected predictions for IDR-IDR and IDR-any interactions.
			preds_dict["AF2_IDR-IDR"] = torch.from_numpy( ood_dict["AF2_pLDDT_PAE"]*ood_dict["disorder_mat1"] )
			preds_dict["AF3_IDR-IDR"] = torch.from_numpy( ood_dict["AF3_pLDDT_PAE"]*ood_dict["disorder_mat1"] )
			if "interaction" in task:
				preds_dict["AF2_IDR-any"] = torch.from_numpy( ood_dict["AF2_pLDDT_PAE"]*ood_dict["disorder_mat2"] )
				preds_dict["AF3_IDR-any"] = torch.from_numpy( ood_dict["AF3_pLDDT_PAE"]*ood_dict["disorder_mat2"] )
			
			# Disobind predictions for IDR-IDR and IDR-any interactions.
			preds_dict["Disobind_uncal_IDR-IDR"] = torch.from_numpy( ood_dict["Disobind_uncal"]*ood_dict["disorder_mat1"] )
			preds_dict["Disobind_cal_IDR-IDR"] = torch.from_numpy( ood_dict["Disobind_cal"]*ood_dict["disorder_mat1"] )
			if "interaction" in task:
				preds_dict["Disobind_uncal_IDR-any"] = torch.from_numpy( ood_dict["Disobind_uncal"]*ood_dict["disorder_mat2"] )
				preds_dict["Disobind_cal_IDR-any"] = torch.from_numpy( ood_dict["Disobind_cal"]*ood_dict["disorder_mat2"] )

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
				# Create contact density vs interface 1 prediction plots.
				self.create_performance_contactdensity_plots( entry_ids, 
																preds_dict["Disobind_cal"], 
																preds_dict["AF2_pLDDT_PAE"], 
																ood_dict["targets"] )

			# Do for all tasks.
			self.create_calibration_plots( preds_dict["Disobind_uncal"], preds_dict["Disobind_cal"], ood_dict["targets"], task )

			# Just adding a empty row for separating different tasks.
			for key in results_dict.keys():
				results_dict[key].append( "" )

		self.create_sparsity_f1_plots( results_dict )
		self.case_specific_analysis()

		# Dump all calculated metrics on disk.
		df = pd.DataFrame( results_dict )
		df.to_csv( self.full_results_file )

		# Dump a subset of results o disk - for ease of looking.
		subset_dict = {key:[] for key in results_dict.keys() if key not in ["AvgPrecision", "MCC", "AUROC", "Accuracy"]}
		for i in range( len( results_dict["Model"] ) ):
			if results_dict["Model"][i] in ["AF2_pLDDT_PAE", "AF3_pLDDT_PAE","Disobind_cal", 
											"AF2_Disobind_cal", "AF3_Disobind_cal",
											"AF2_IDR-IDR", "AF3_IDR-IDR", "Disobind_cal_IDR-IDR", "Disobind_cal_IDR-any", ""]:
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



	def create_performance_contactdensity_plots( self, entries, diso_preds, af2_preds, target ):
		"""
		Create a plot for the performance(Recall/Precision/F1) vs contact density for Interface_1.
		Save the raw data on disk.
		Also save the top predictions (> 0.7 F1 score) on disk.

		Input:
		----------
		entries --> list of all entry_id for all OOD entries.
		diso_preds --> (torch.tensor) Calibrated Disobind prediction.
		af2_preds --> (torch.tensor) AF2 predictions.
		target --> (torch.tensor) binary output labels.

		Returns:
		----------
		None
		"""
		entries = np.array( entries )

		# Get per OOD entry metrics.
		metrics_diso = torch_metrics( preds = diso_preds.to( self.device ), 
								target = torch.from_numpy( target ).to( self.device ), 
								threshold = self.contact_threshold,
								multidim_avg = "samplewise_none", device = self.device )
		

		metrics_af2 = torch_metrics( preds = af2_preds.to( self.device ), 
								target = torch.from_numpy( target ).to( self.device ), 
								threshold = self.contact_threshold,
								multidim_avg = "samplewise_none", device = self.device )
		
		contact_density = []
		for i, id_ in enumerate( entries ):
			head1, head2 = id_.split( "--" )
			head2, num = head2.split( "_" )
			uni_id1, r11, r12 = head1.split( ":" )
			uni_id2, r21, r22 = head2.split( ":" )
			len1 = int( r12 ) - int( r11 ) + 1
			len2 = int( r22 ) - int( r21 ) + 1
			
			contacts = np.count_nonzero( target[i] )
			num_elements = len1 + len2

			contact_density.append( contacts/num_elements )

		contact_density = np.array( contact_density ).reshape( 60, 1 )

		re_diso = metrics_diso[0].cpu().numpy()
		prec_diso = metrics_diso[1].cpu().numpy()
		f1_diso = metrics_diso[2].cpu().numpy()

		re_af2 = metrics_af2[0].cpu().numpy()
		prec_af2 = metrics_af2[1].cpu().numpy()
		f1_af2 = metrics_af2[2].cpu().numpy()

		fig, ax = plt.subplots( 1, 3, figsize = ( 20, 8 ) )
		ax[0].scatter( contact_density, re_diso )
		ax[0].set_ylabel( "Recall" )
		ax[0].set_xlabel( "Contact density" )

		ax[1].scatter( contact_density, prec_diso )
		ax[1].set_ylabel( "Precision" )
		ax[1].set_xlabel( "Contact density" )
		
		ax[2].scatter( contact_density, f1_diso )
		ax[2].set_ylabel( "F1 score" )
		ax[2].set_xlabel( "Contact density" )

		plt.savefig( f"{self.contact_density_file}.png", dpi = 300 )
		plt.close()

		# Save samplewsie metrics for Disobind and AF2 on OOD set.
		df = pd.DataFrame()
		df["Entry ID"] = entries
		df["Contact density"] = np.round( contact_density.reshape( -1 ), 3 )
		df["Recall"] = np.round( list( re_diso ), 3 )
		df["Precision"] = np.round( list( prec_diso ), 3 )
		df["F1"] = np.round( list( f1_diso ), 3 )
		df["AF2-Recall"] = np.round( list( re_af2 ), 3 )
		df["AF2-Precision"] = np.round( list( prec_af2 ), 3 )
		df["AF2-F1"] = np.round( list( f1_af2 ), 3 )
		df.to_csv( f"{self.contact_density_file}.csv" )

		# Save samplewsie metrics for Disobind and AF2 on best performing pairs.
		idx = np.where( f1_diso > 0.7 )
		top = entries[idx]

		df = pd.DataFrame()
		df["Entry ID"] = top
		df["Contact density"] = np.round( contact_density[idx], 3 )
		df["Recall"] = np.round( re_diso[idx], 3 )
		df["Precision"] = np.round( prec_diso[idx], 3 )
		df["F1 score"] = np.round( f1_diso[idx], 3 )
		df["AF2-Recall"] = np.round( re_af2[idx], 3 )
		df["AF2-Precision"] = np.round( prec_af2[idx], 3 )
		df["AF2-F1 score"] = np.round( f1_af2[idx], 3 )
		df.to_csv( f"{self.top_preds_file}.csv" )

		plt.scatter( f1_diso[idx], f1_af2[idx], color = "blue" )
		plt.xlim( 0.5, 1 )
		plt.ylim( -0.1, 1 )
		plt.savefig( f"{self.top_preds_diso_af2_file}.png", dpi = 300 )
		plt.close()



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
			if results_dict["Model"][i] == "Disobind_cal":
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
		top_pairs = ["P04273:95:193--P04273:95:193_0", "P07101:109:160--P07101:109:160_1",
					"P58771:132:192--P58771:132:192_11", "P12883:1785:1850--P12883:1785:1850_206",
					"P58771:181:270--P58771:181:270_17", "P58771:71:131--P58771:71:131_2",
					"Q91Z83:1875:1935--Q91Z83:1875:1935_146", "Q16236:506:559--O15525:23:122_0"]

		fig, ax = plt.subplots( 8, 1, figsize = ( 15, 15 ) ) # , constrained_layout = True
		# fig.tight_layout()

		i = 0
		for entry_id in top_pairs:
			uni_id1, uni_id2 = entry_id.split( "--" )
			uni_id2, cp = uni_id2.split( "_" )
			uni_id1, _, _ = uni_id1.split( ":" )
			uni_id2, _, _ = uni_id2.split( ":" )

			id_ = f"{uni_id1}--{uni_id2}_{cp}"

			diso_pred = self.disobind_preds[entry_id]["interface_1"]["Disobind_cal"]
			diso_pred = np.where( diso_pred >= self.contact_threshold, 1, 0 )

			hf = h5py.File( f"{self.merged_binary_complexes_dir}{id_}.h5" )
			print( id_ )
			print( np.array( hf["merged_entries"] ) )
			total = np.array( hf["conformers"] )
			summed_cmap = np.array( hf["summed_cmap"] )/total
			
			pad = np.zeros( ( self.max_len, self.max_len ) )
			m, n = summed_cmap.shape
			pad[:m, :n] = summed_cmap
			summed_cmap = pad
			# m = nn.MaxPool2d(  )

			merged_interface = np.concatenate( 
											( np.max( summed_cmap, axis = 1 ),
											np.max( summed_cmap, axis = 0 ) ),
											axis = 0
											 ).reshape( 200, 1 )
			empty = np.zeros( ( 200, 1 ) )
			
			plot = np.stack( [diso_pred, empty, merged_interface ] )
			ax[i].imshow( plot, cmap = plt.cm.Greens.reversed() )
			ax[i].set_aspect( 8 )
			plt.subplots_adjust( hspace = 0 )
			i += 1

		plt.savefig( f"{self.top_preds_file}.png", dpi = 300 )
		plt.close()



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

		return np.round( metric_array, 3 )


if __name__ == "__main__":
	JudgementDay().forward()
	print( "May the Force be with you..." )
