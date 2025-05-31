"""
Run Disobind for all protein pairs selected from IDPPI test set.
"""
import os, json, time, subprocess
from typing import List, Dict
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import torch

from dataset.create_input_embeddings import Embeddings
from src.models.get_model import get_model
from src.metrics import torch_metrics
from params import parameter_files


class IdppiPreds():
	"""
	Get Disobind for all protein pairs selected from IDPPI test set.
	Evaluate performance of Disobind.
	"""
	def __init__( self ):
		self.base_dir = os.path.abspath( "./" )
		self.data_dir = "../database/idppi/"
		self.output_dir = "./idppi_preds/"
		
		self.idppi_input_file = os.path.join( self.data_dir, "IDPPI_input_diso.csv" )
		self.idppi_targets_file = os.path.join( self.data_dir, "IDPPI_target.json" )

		self.model_version = 19
		self.batch_size = 250
		self.max_len = 200
		# Threshold to define a contact from Disobind output.
		self.contact_threshold = 0.5
		self.device = "cuda"
		self.scope = "global" # global/local
		self.embedding_type = "T5"

		self.predictions = {}

		# Load a dict storing paths for each model.
		self.parameters = parameter_files( self.model_version )

		self.uniprot_seq_file = os.path.join( self.data_dir, "Uniprot_seq_idppi.json" )
		self.emb_fasta_file_prefix = "idppi_emb_fasta"
		# os.path.join( os.path.abspath( self.output_dir ),
		# 									"idppi_emb_fasta.fasta" )
		self.emb_file_prefix = "idppi_embeddings"
		# os.path.join( os.path.abspath( self.output_dir ),
		# 									"idppi_embeddings.h5" )
		self.predictions_file = os.path.join( self.output_dir, "Disobind_Predictions_idppi.npy" )
		# self.predictions_file = "/data2/kartik/Disorder_Proteins/disobind/analysis/idppi_preds/Disobind_Predictions_idppi.npy"
		self.output_dict_file = os.path.join( self.output_dir, "Results_IDPPI.csv" )


	def forward( self ):
		if os.path.exists( self.predictions_file ):
			print( "Disobind predictions for IDPPI already exist..." )
			self.predictions = np.load( self.predictions_file, allow_pickle = True ).item()
		else:
			tic = time.time()
			self.create_dirs()
			with open( self.uniprot_seq_file, "r" ) as f:
				self.uniprot_seq_dict = json.load( f )
			entry_ids = self.read_csv()
			self.get_predictions( entry_ids )
			np.save( self.predictions_file, self.predictions, allow_pickle = True )
			toc = time.time()
			print( f"Total time taken for Disobind prediction = {( toc-tic )/60} minutes" )

		output_dict = self.analysis_idppi()

		self.save_results( output_dict )

		print( "May the Force be with you..." )


	def create_dirs( self ):
		"""
		Create the required dirs.
		"""
		os.makedirs( self.output_dir, exist_ok = True )


	def read_csv( self ):
		"""
		Parse the .csv file containing entry_ids for IDPPI dataset.
		"""
		with open( self.idppi_input_file, "r" ) as f:
			entry_ids = f.readlines()[0].split( "," )
		return entry_ids


	def split_entry_id( self, entry_id: str, return_pos = False ):
		"""
		Split the entry_id into individual components.
		"""
		uni_id1, uni_id2 = entry_id.split( "--" )
		uni_id2, copy_num = uni_id2.split( "_" )
		uni_id1, start1, end1 = uni_id1.split( ":" )
		uni_id2, start2, end2 = uni_id2.split( ":" )
		if return_pos:
			return uni_id1, int( start1 ), int( end1 ), uni_id2, int( start2 ), int( end2 ), copy_num
		else:
			return uni_id1, uni_id2, copy_num


	def get_pair_id_from_entry_id( self, entry_id: str ):
		"""
		Given the entry_id, return the pair_id.
		"""
		uni_id1, uni_id2, copy_num = self.split_entry_id( entry_id, return_pos = False )
		pair_id = f"{uni_id1}--{uni_id2}"
		return pair_id


	def create_embeddings( self, entry_ids: List[str], emb_fasta_file: str, emb_file: str ):
		"""
		Use the Embeddings() class to:
			Create fasta files and get embeddings.
		"""
		prot1_emb, prot2_emb = Embeddings( scope = self.scope,
											embedding_type = self.embedding_type, 
											uniprot_seq = self.uniprot_seq_dict,
											base_path = self.output_dir, 
											fasta_file = emb_fasta_file, 
											emb_file = emb_file, 
											headers = entry_ids, 
											load_cmap = False
											 ).initialize( return_emb = True )
		return prot1_emb, prot2_emb


	################################################################################
	################################################################################
	def get_predictions( self, entry_ids ):
		"""
		Run interface_1 prediction for all entry_ids in batches.
		"""
		model = self.load_model()
		total_pairs = len( entry_ids )
		for start in np.arange( 0, total_pairs, self.batch_size ):
			t_start = time.time()
			if start + self.batch_size >= total_pairs:
				end = total_pairs
			else:
				end = start + self.batch_size
			
			print( f"\nBatch {start}:{end}/{total_pairs}" +
						"-----------------------------------------" )
			batch = entry_ids[start:end]

			print( f"Running predictions for batch {start}-{end}..." )
			emb_fasta_file = os.path.join( os.path.abspath( self.output_dir ),
											f"{self.emb_fasta_file_prefix}_{start}-{end}.fasta" )
			emb_file = os.path.join( os.path.abspath( self.output_dir ),
											f"{self.emb_file_prefix}_{start}-{end}.h5" )
			batch_preds = self.get_preds_for_batch( model, batch, emb_fasta_file, emb_file )
			self.predictions.update( batch_preds )
			# self.update_predictions_dict( batch_preds )

			t_end = time.time()
			print( f"Time taken for batch {start}-{end} = {( t_end - t_start )/60} minutes\n" )
			print( f"\nCompleted for batch {start}:{end}/{total_pairs}" +
					"-----------------------------------------" )

			# subprocess.call( ["rm", f"{self.emb_file}", f"{self.fasta_file}"] )


	def get_preds_for_batch( self, model, batch: List[str], emb_fasta_file: str, emb_file: str ):
		"""
		Create embeddings and run interface_1 predictions for the
			given batch of entry_ids.
		"""
		print( "Creating global embeddings for the input sequences..." )
		prot1_emb_dict, prot2_emb_dict = self.create_embeddings( batch, emb_fasta_file, emb_file )

		os.chdir( self.base_dir )

		print( "Running predictions..." )
		batch_preds = self.predict( model, prot1_emb_dict, prot2_emb_dict )

		# Release all unoccupied cached memory currently held by the caching allocator.
		torch.cuda.empty_cache()
		time.sleep( 5 )

		subprocess.call( ["rm", f"{emb_file}", f"{emb_fasta_file}"] )
		return batch_preds


	# def update_predictions_dict( self, batch_preds: Dict ):
	# 	"""
	# 	Update predictions for all entry_ids to the respective pair_id.
	# 	"""
	# 	for pair_id in batch_preds:
	# 		if pair_id not in self.predictions:
	# 			self.predictions[pair_id] = {}
	# 		for entry_id in batch_preds[pair_id]:
	# 			self.predictions[pair_id][entry_id] = batch_preds[pair_id][entry_id].copy()

	################################################################################
	################################################################################
	def load_model( self ):
		"""
		Load pre-trained model in evaluation mode for making predictions.
		"""
		mod_ver = self.parameters["interface"]["cg_1"][0]
		model_config = f"../params/Model_config_{mod_ver}.yml"
		model_config = OmegaConf.load( model_config )
		model_config = model_config.conf.model_params

		# mod_ver = mod_ver.split( "_" )
		# Model name.
		m = mod_ver.split( "_" )
		mod = "_".join( m[:-1] )
		# model version.
		ver = m[-1]
		params_file = self.parameters["interface"]["cg_1"][1]
		# Load Disobind model.
		model = get_model( model_config ).to( self.device )
		model.load_state_dict( 
							torch.load( f"../models/{mod}/Version_{ver}/{params_file}.pth", 
										map_location = self.device )
							 )
		model.eval()

		return model


	def prepare_emb_tensor( self, prot_emb: np.array ) -> torch.Tensor:
		"""
		Given the input prot embedding:
			Pad to max_len.
			Convert to torch.Tensor.
		"""
		# prot1 --> [L,C]; K -> protein length, C -> feature dim.
		# m, n = prot_emb.shape
		# mask = np.zeros( [self.max_len, n] )

		# mask[:m,:] = prot_emb
		# prot_emb = mask
		prot_emb = torch.from_numpy( prot_emb
							).unsqueeze( 0 ).float()
		return prot_emb.to( self.device )


	def create_output_mask( self, p1_len: int, p2_len: int ) -> np.array:
		"""
		Given lengths for prot1/2, create a binary output mask for
			interface_1 prediction.
		"""
		p1_mask = np.zeros( [self.max_len] )
		p1_mask[:p1_len] = 1
		p2_mask = np.zeros( [self.max_len] )
		p2_mask[:p2_len] = 1
		output_mask = np.concatenate( [p1_mask,p2_mask], axis = 0 )
		return output_mask


	def predict( self, model, prot1_emb_dict: Dict, prot2_emb_dict: Dict ):
		"""
		Obtain Disobind interface_1 predictions.
		For all entry_ids provided,
			Pad the prot1/2 embeddings to max_len.
			Get model prediction.
		Predictions are stored as:
			preds_dict ={
				"pair_id": {
					"entry_id": 
				}
			}
		"""
		preds_dict = {}
		for entry_id in prot1_emb_dict:
			# pair_id = self.get_pair_id_from_entry_id( entry_id )
			# if pair_id not in preds_dict:
			# 	preds_dict[pair_id] = {}

			p1_emb = prot1_emb_dict[entry_id]
			p2_emb = prot2_emb_dict[entry_id]

			p1_len = p1_emb.shape[0]
			p2_len = p2_emb.shape[0]

			prot1_emb = self.prepare_emb_tensor( p1_emb )
			prot2_emb = self.prepare_emb_tensor( p2_emb )

			with torch.no_grad():
				pred_interface = model( prot1_emb, prot2_emb )
				pred_interface = pred_interface.squeeze( 0 ).detach().cpu().numpy()
				# output_mask = self.create_output_mask( p1_len, p2_len )
				# pred_interface = pred_interface*output_mask
			preds_dict[entry_id] = pred_interface
		return preds_dict

	################################################################################
	################################################################################
	def analysis_idppi( self ):
		"""
		Given Disobind interface_1 predictions, determine
			if the protein pair interacts or not (PPI task).
		"""
		print( "\nIDPPI analysis..." )
		output_dict = {k:[] for k in ["test_name", "contact_threshold", "Recall", "Precision",
										"F1score", "AvgPrecision", "MCC", "AUROC", "Accuracy"]}
		with open( self.idppi_targets_file, "r" ) as f:
			targets_dict = json.load( f )

		for ct in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]:
		# for ct in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9999]:
			self.contact_threshold = ct
			print( f"Contact threshold = {self.contact_threshold}..." )

			ppi_dict = self.get_ppi_dict()
			preds_dict, targets = self.prepare_pred_target_tensors( ppi_dict, targets_dict )

			for test_name in preds_dict:
				print( f"{test_name} ---------" )
				metrics = self.calculate_metrics( preds_dict[test_name], targets )

				output_dict["test_name"].append( test_name )
				output_dict["contact_threshold"].append( ct )
				for i, metric in enumerate( ["Recall", "Precision", "F1score", "AvgPrecision",
												"MCC", "AUROC", "Accuracy"] ):
					output_dict[metric].append( metrics[i] )
				print( metrics )
			for k in output_dict:
				output_dict[k].append( "" )
			print( "\n------------------------------------\n" )
		return output_dict


	def prepare_pred_target_tensors( self, ppi_dict: Dict, targets_dict: Dict ):
		"""
		Create a pred and target tensor for metric calculation.
		"""
		# pred, target = [], []
		preds_dict = {k:[] for k in ppi_dict.keys()}
		targets = []
		# Flag to parse targets_dict only once.
		no_target = False

		for test_name in ppi_dict:
			for entry_id in ppi_dict[test_name]:
				pair_id = self.get_pair_id_from_entry_id( entry_id )

				preds_dict[test_name].append( ppi_dict[test_name][entry_id] )
				if not no_target:
					targets.append( int( targets_dict[pair_id] ) )
			no_target = True
			preds_dict[test_name] = torch.from_numpy( np.array( preds_dict[test_name]
													) ).float().to( self.device )
		
		targets = torch.from_numpy( np.array( targets ) ).float().to( self.device )

		return preds_dict, targets


	def get_ppi_dict( self ):
		"""
		Test different ways to obtain PPI predictions.
		Store results for each test in ppi_dict.
		"""
		ppi_dict = {}
		for entry_id in self.predictions:
			for test in [self.interface_to_ppi_1, self.interface_to_ppi_2,
						self.interface_to_ppi_3, self.interface_to_ppi_4,
						self.interface_to_ppi_5, self.interface_to_ppi_6]:
				test_name, ppi_pred = test( entry_id )
				if test_name not in ppi_dict:
					ppi_dict[test_name] = {entry_id: ppi_pred}
				else:
					ppi_dict[test_name][entry_id] = ppi_pred
		return ppi_dict


	def interface_to_ppi_1( self, entry_id: str ):
		"""
		'prot_any'
		A protein pair is interacting if,
			any( prot1, prot2 )
			have predicted interface residues.
		"""
		name = "prot_any"

		pred_interface = self.predictions[entry_id]
		pred_interface = np.where( pred_interface > self.contact_threshold, 1, 0 )
		if np.count_nonzero( pred_interface ) > 0:
			ppi_pred = 1
		else:
			ppi_pred = 0
		return name, ppi_pred


	def interface_to_ppi_2( self, entry_id: str ):
		"""
		'prot_all'
		A protein pair is interacting if,
			all( prot1, prot2 )
			have predicted interface residues.
		"""
		name = "prot_all"

		pred_interface = self.predictions[entry_id]
		pred_interface = np.where( pred_interface > self.contact_threshold, 1, 0 )

		( uni_id1, start1, end1,
			uni_id2, start2, end2, copy_num ) = self.split_entry_id( entry_id, return_pos = True )
		p1_interface = pred_interface[:end1]
		p2_interface = pred_interface[end1:]

		if np.count_nonzero( p1_interface ) > 0 and np.count_nonzero( p2_interface ) > 0:
			ppi_pred = 1
		else:
			ppi_pred = 0
		return name, ppi_pred


	def interface_to_ppi_3( self, entry_id: str ):
		"""
		'max_score_any'
		A protein pair is interacting if, max score
			across all residues in both protein > contact threshold.
		"""
		name = "max_score_any"

		pred_interface = self.predictions[entry_id]
		ppi_pred = np.max( pred_interface )

		return name, ppi_pred


	def interface_to_ppi_4( self, entry_id: str ):
		"""
		'avg_score_any'
		A protein pair is interacting if, the avg score
			across all residues for both protein is > contact threshold.
		"""
		name = "avg_score_any"
		frag_pred = []
		uni_id1, start1, end1, uni_id2, start2, end2, copy_num = self.split_entry_id( entry_id, return_pos = True )
		pred_interface = self.predictions[entry_id]

		ppi_pred = np.mean( pred_interface )
		return name, ppi_pred


	def interface_to_ppi_5( self, entry_id: str ):
		"""
		'avg_score_all'
		A protein pair is interacting if, the avg of tghe avg score
			across all residues for each protein is > contact threshold.
		"""
		name = "avg_score_all"
		frag_pred = []
		uni_id1, start1, end1, uni_id2, start2, end2, copy_num = self.split_entry_id( entry_id, return_pos = True )
		pred_interface = self.predictions[entry_id]
		p1_interface = pred_interface[:end1]
		p2_interface = pred_interface[end1:]


		ppi_pred = ( np.mean( p1_interface ) + np.mean( p2_interface ) )/2
		return name, ppi_pred


	def interface_to_ppi_6( self, entry_id: str ):
		"""
		'avg_max_score'
		A protein pair is interacting if, the avg of the max score
			across all residues for each proteins is > contact threshold.
		"""
		name = "avg_max_score"
		frag_pred = []
		uni_id1, start1, end1, uni_id2, start2, end2, copy_num = self.split_entry_id( entry_id, return_pos = True )
		pred_interface = self.predictions[entry_id]

		p1_interface = pred_interface[:end1]
		p2_interface = pred_interface[end1:]

		ppi_pred = ( np.max( p1_interface ) + np.max( p2_interface ) )/2
		return name, ppi_pred


	################################################################################
	################################################################################
	def calculate_metrics( self, pred: torch.Tensor, target: torch.Tensor ):
		"""
		Calculate the following metrics:
			Recall, Precision, F1score, AvgPrecision, MCC, AUROC, Accuracy.
		metric_array --> np.array containing the calculated metric values in order:
			Recall, Precision, F1score, AvgPrecision, MCC, AUROC, Accuracy.	
		"""
		metrics = torch_metrics( pred, target, self.contact_threshold, "global", self.device )
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


	################################################################################
	################################################################################
	def save_results( self, output_dict: Dict[str, List] ):
		"""
		Save the results dict as a .csv file.
		"""
		df = pd.DataFrame( output_dict )
		df.to_csv( self.output_dict_file, index = False )


if __name__ == "__main__":
	IdppiPreds().forward()

