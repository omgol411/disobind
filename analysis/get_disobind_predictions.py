######## Make predictions using trained Disobind models. #######
######## ------>"May the Force serve u well..." <------#########
################################################################

############# One above all #############
##-------------------------------------##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import os
import argparse
import h5py
import json
import joblib

import torch
from torch import nn

import dataset
from dataset.from_APIs_with_love import ( get_uniprot_seq, get_motifs_from_elm )
from dataset.utility import ( consolidate_regions, find_disorder_regions, load_disorder_dbs )
from dataset.create_input_embeddings import Embeddings
from src.models.get_model import get_model
from src.metrics import torch_metrics
from src.utils import prepare_input
from params import parameter_files

"""
Make predictions using the trained model(s).
The input could be wither:
	A csv file containg the Prot1 and Prot2 as: UniID1:start1:end1--UniID2:start2:end2_num
	A fasta file containing both Prot1 nad prot2 with the headers as: >UniID1:start1:end1.
The former can be used to run predictions in batch while the latter would run only one pair at a rime.
In case no Uniprot ID is available for a protein, use the fasta file format.
	Keep the header as: >Prot1:start1:end1.
	Provide full protein sequence.
"""

class Prediction():
	def __init__( self ):
		"""
		Constructor
		"""
		# Input file type - csv/fasta.
		self.input_type = "csv"
		# Input file containing the prot1/2 headers.
		self.input_file = "../database/v_19/prot_1-2_test_v_19.csv"
		# Embedding type to be used for prediction.
		self.embedding_type = "T5"
		# Use global/local embeddings.
		self.scope = "global"
		self.device = "cuda" # cpu/cuda.
		# Max protein length.
		self.max_len = 100
		# Contact probability threshold.
		self.threshold = 0.5
		self.multidim_avg = "global" # global/samplewise/samplewise-none.
		# Objective settings to be used for prediction.
		self.objective = ["", "", "", ""]
		# Load a dict storing paths for each model.
		self.parameters = parameter_files()
		# Dict to store predictions for all tasks.
		self.predictions = {}
		self.counts = [[], [], []]

		# Dict to store Uniprot sequences.
		self.uniprot_seq = {}
		# Dict to store disordered residues for all UniProt IDs.
		self.disorder_dict = {}

		# Csv file path for DisProt/IDEAL?mobiDB.
		self.disprot_path = f"../database/input_files/DisProt.csv"
		self.ideal_path = f"../database/input_files/IDEAL.csv"
		self.mobidb_path = f"../database/input_files/MobiDB.csv"

		# Test contact maps file name.
		self.cmaps_file =  f"../database/v_{self.version}/Output_bcmap_test_v_{self.version}.h5"
		# Uniprot file name.
		self.Uniprot_seq_file =  f"../database/v_{self.version}/Uniprot_seq.json"
		# self.output_dir = f"Predictions_{self.mod}_{self.mod_ver}"
		# Name for output directory.
		if self.mode == "ood":
			self.output_dir = f"Predictions_ood_v{self.version}"
			# Filename to store predictions.
			self.output_filename = "Disobind_Predictions.npy"
		elif self.mode == "ped":
			self.output_dir = f"Predictions_ped"
			# Filename to store predictions.
			self.output_filename = "PED_Predictions.npy"
		else:
			self.output_dir = f"Disobind_Predictions"
			# Filename to store predictions.
			self.output_filename = "Predictions.npy"		
		if os.path.exists( self.output_dir ):
			reply = input( "Output directory already exists. Abort? (Y/n)\t" )
			if reply == "Y":
				exit()
		else:
			os.makedirs( self.output_dir, exist_ok = True )
		
		# Absolute path to the analysis dir.
		self.abs_path = os.path.abspath( "disobind/analysis/" )
		# Path fo the FASTA file for prot1/2 sequences.
		self.fasta_file = f"{self.abs_path}/{self.output_dir}/p1_p2_test.fasta"
		# Path fo the h5 file for prot1/2 embeddings.
		self.emb_file = f"{self.abs_path}/{self.output_dir}/p1_p2_test.h5"

		# Disordered residues dict file path.
		self.disorder_file_path = f"{self.abs_path}/{self.output_dir}/disorder_dict.json"
		# Logs file path.
		self.logs_file = f"{self.abs_path}/{self.output_dir}/Logs.json"

		self.logs = {"counts": {}}


	def forward( self ):
		"""
		Get the Uniprot sequences for all unique Uniprot IDs amongst prot1/2.
			For test mode,
				Use the downloaded Uniprot sequences.
			For predict mode,
				Download the sequences.
		"""
		self.disprot, self.ideal, self.mobidb = load_disorder_dbs( disprot_path = self.disprot_path, 
																	ideal_path = self.ideal_path, 
																	mobidb_path = self.mobidb_path )

		headers = self.read_input()

		with open( self.Uniprot_seq_file, "r" ) as f:
			self.uniprot_seq = json.load( f )
		# Contact maps.
		self.cmaps = h5py.File( self.cmaps_file, "r" )

		if os.path.exists( self.disorder_file_path ):
			with open( self.disorder_file_path, "r" ) as f:
				self.disorder_dict = json.load( f )

		print( "Creating global embeddings for the input sequences..." )
		self.create_embeddings( headers )

		os.chdir( self.abs_path )

		print( "Running predictions..." )
		self.predict()

		np.save( f"./{self.output_dir}/{self.output_filename}", self.predictions )
		
		with open( self.logs_file, "w" ) as w:
			json.dump( self.logs, w )


	def read_input( self ):
		"""
		Read the input file and return the headers for getting the embeddings.
			Only .csv and .fasta formats are supported.
		.csv format is preferable for predictions on multiple binary complexes.
		.fasta format does not support having >1 binary complex in a file.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		headers --> list of entry_id for a binary complexes.
		"""
		if self.input_type == "csv":
			headers = self.read_csv()

		elif self.input_type == "fasta":
			headers = self.read_fasta()

		else:
			raise Exception( "Incorrect file format specified.\n Only .csv and .fasta format are supported." )

		# headers_ = []
		# Just remove any empty element e.g. "".
		headers = [head for head in headers if len( head ) != 0]
		return headers


	def read_csv( self ):
		"""
		Read from a csv file containing prot1 and prot2 info as:
			UniID1:start1:end1--UniID2:start2:end2_num,

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		headers --> list of entry_id for a binary complexes.
		"""
		with open( self.input_file, "r" ) as f:
			headers = f.readlines()[0].split( "," )
		return headers


	def read_fasta( self ):
		"""
		Read from a fasta file containing prot1 and prot2 info as:
			>UniID1:start1:end1
			Sequence1
			UniID2:start2:end2
			>Sequence2
		Provide the complete Uniprot sequence for both the proteins.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		headers --> list of entry_id for a binary complexes.
		"""
		headers = []
		with open( self.input_file, "r" ) as f:
			headers.append( f.readlines()[0].split( ">" )[1] )
			headers.append( f.readlines()[2].split( ">" )[1] )
		return headers


	def create_embeddings( self, headers ):
		"""
		Use the Shredder() class to:
			Create fasta files and get embeddings.

		Input:
		----------
		headers --> list of entry_id for a binary complexes.

		Returns:
		----------
		None
		"""
		self.prot1_emb, self.prot2_emb = Embeddings( scope = self.scope, embedding_type = self.embedding_type, 
													uniprot_seq = self.uniprot_seq, base_path = self.output_dir, 
													fasta_file = self.fasta_file, 
													emb_file = self.emb_file, 
													headers = headers, 
													load_cmap = False
													 ).initialize( return_emb = True )


	def load_model( self, model_ver, params_file ):
		"""
		Load pre-trained model in evaluation mode for making predictions.

		Input:
		----------
		model_ver --> version of the model to be loaded.
		params_file --> model parameters file.

		Returns:
		----------
		model --> trained model in eval() mode.
		"""
		model_config = f"../params/Model_config_{model_ver}.yml"
		model_config = OmegaConf.load( model_config )
		model_config = model_config.conf.model_params

		model_ver = model_ver.split( "_" )
		mod = "_".join( model_ver[:-1] )
		ver = model_ver[-1]
		model = get_model( model_config ).to( self.device )
		model.load_state_dict( 
							torch.load( f"../models/{mod}/Version_{ver}/{params_file}.pth", 
										map_location = self.device )
							 )
		model.eval()


		return model


	def apply_settings( self, obj, cg ):
		"""
		Load models and modify the self.objective depending on the model type.
			self.objective includes 4 hparams to be set:
			0. objective: interaction or interface
			1. bin_size: level of coarse graining.
			2. bin_input: if bin_size > 1, average the input embeddings.
			3. single_output: if True, use single output prediction task.

		Input:
		----------
		key1 --> specifies the model to be used.
		cg --> specifies the coarse graining to be used.

		Returns:
		----------
		model --> model corresponding to the specified settings. 
		"""
		print( f"\nLoading model for {obj}: CG = {cg}..." )
		model = self.load_model( *self.parameters[obj][cg] )

		if "interaction" in obj:
			self.objective[0] = "interaction"
			if "mo" in obj:
				self.objective[3] = False
			elif "so" in obj:
				self.objective[3] = True
		
		elif "interface" in obj:
			self.objective[0] = "interface"
			self.objective[3] = False

		_, cg = cg.split( "_" )
		self.objective[1] = int( cg )
		if int( cg ) != 1:
			self.objective[2] = True
		else:
			self.objective[2] = False
		return model


	def get_input_tensors( self, key, prot1, prot2, target ):
		"""
		Prepare prot1, prot2, target tensors according to the objective.
			Pad the prot1/2 and cmap arrys.
			Create a binary target_mask.
		Convert the input arrays to torch.tensor and load on the specified device.
		**Note: If not using padding, terminal residues might be lost upon coarse graining 
				if protein length not divisible by bin_size.

		Input:
		----------
		prot1 --> (np.array) prot1 embedding of dimension [L1, C].
		prot1 --> (np.array) prot2 embedding of dimension [L2, C].
		target --> (np.array) contact map of dimension [L1, L2].

		Returns:
		----------
		prot1 --> (torch.tensor) prot1 embedding of dimension [N, L1, C].
		prot2 --> (torch.tensor) prot2 embedding of dimension [N, L2, C].
		target --> (torch.tensor) contact map of dimension [N, L1, L2].
			If padding, L1 == L2; N = 1.
		target_mask --> (torch.tensor) binary mask of dimension [N, L1, L2].
		eff_len --> effective length of prot1/2 (eff_len) post coarse graining.
		"""		
		num_res1 = prot1.shape[0]
		num_res2 = prot2.shape[0]
		# print( num_res1, "  ", num_res2, "  ", prot1.shape, "  ", key )

		if prot1.shape[0] != target.shape[0] or prot2.shape[0] != target.shape[1]:
			raise Exception( f"Mismatch in prot1/2 lengths and cmap dimension. {prot1.shape}  {prot2.shape}  {target.shape}..." )

		self.counts[0].append( np.count_nonzero( target ) )
		self.counts[1].append( num_res1 )
		self.counts[2].append( num_res2 )

		eff_len = self.max_len//self.objective[1]
		eff_len = [eff_len, eff_len]
		
		mask1 = np.zeros( ( self.max_len, 1024 ) )
		mask2 = np.zeros( ( self.max_len, 1024 ) )

		mask1[:num_res1, :] = prot1
		mask2[:num_res2, :] = prot2

		prot1 = mask1
		prot2 = mask2

		target_mask = np.zeros( ( 1, self.max_len, self.max_len ) )
		target_mask[:, :num_res1,:num_res2] = 1
		target_padded = np.copy( target_mask )
		target_padded[:, :num_res1,:num_res2] = target
		target = target_padded

		target = torch.from_numpy( target )
		target_mask = torch.from_numpy( target_mask )

		prot1 = torch.from_numpy( prot1 ).unsqueeze( 0 )
		prot2 = torch.from_numpy( prot2 ).unsqueeze( 0 )

		prot1, prot2, target, target_mask = prepare_input( prot1, prot2, target, 
															[True, target_mask], 
															objective = self.objective[0], 
															bin_size = self.objective[1], 
															bin_input = self.objective[2], 
															single_output = self.objective[3] )
		prot1 = prot1.to( self.device ).float()
		prot2 = prot2.to( self.device ).float()
		target_mask = target_mask.to( self.device )

		return prot1, prot2, target, target_mask, eff_len



	def get_disordered_positions( self, uni_id ):
		"""
		Obtain all disordered regions for the given Uniprot ID.
		Convert residue ranges to positions for all disordered regions.
		Remove duplicates and Sort.

		Inputs:
		----------
		uni_id --> (str) Uniprot ID.

		Returns:
		----------
		disorder_positions --> list of all diosrdered residue positions for the given Uniprot ID.
		"""
		disorder_regions = find_disorder_regions( disprot = self.disprot, 
														ideal = self.ideal, 
														mobidb = self.mobidb, 
														uni_ids = [uni_id], 
														min_len = 1, return_ids = False )

		# Convert disordered regions to disordered positions.
		disorder_positions = []
		for reg in disorder_regions:
			disorder_positions.extend( reg )
		# Remove duplicate residue positions and sort.
		disorder_positions = list( sorted( set( disorder_positions ) ) )
		disorder_positions = list( map( int, disorder_positions ) )

		return disorder_positions



	def get_disorder_matrix( self, key, obj, cg ):
		"""
		Create binary matrix for indicating IDR interactions with:
			IDRs only.
			IDR or a non-IDR.

		Obtain disordered positions based on the Uniptot ID.
		Create binary vectors for prot1/2 positions corresponding to disordered residues(1).
		Create a matrix for IDR-IDR and IDR-any interactions.
		Coarse grain the matrix as specified.
		Pad the matrices to max_len.

		Inputs:
		----------
		key --> entry_id for the merged binary complex consisting of -
				UniID1:start1:end1--UniID2:start2:end2_num
		cg --> (str) resolution of coarse graining.

		Returns:
		----------
		disorder_mat1 --> binary matrix for IDR-IDR interactions.
		disorder_mat2 --> binary matrix for IDR-any interactions.
		"""
		head1, head2 = key.split( "--" )
		head2, num = head2.split( "_" )
		uni_id1, start1, end1 = head1.split( ":" )
		uni_id2, start2, end2 = head2.split( ":" )

		if not f"{obj}_{cg}" in self.logs["counts"].keys():
			self.logs["counts"][f"{obj}_{cg}"] = {}
			if "interaction" in obj:
				fields = ["IDR-IDR_interactions", "IDR-any_interactions"]
			elif "interface" in obj:
				fields = ["P1_IDR", "P2_IDR"]
			for k in fields:
				self.logs["counts"][f"{obj}_{cg}"][k] = 0


		if uni_id1 in self.disorder_dict:
			disorder_positions1 = self.disorder_dict[uni_id1]
		else:
			disorder_positions1 = self.get_disordered_positions( uni_id1 )
			self.disorder_dict[uni_id1] = disorder_positions1
		
		if uni_id2 in self.disorder_dict:
			disorder_positions2 = self.disorder_dict[uni_id2]
		else:
			disorder_positions2 = self.get_disordered_positions( uni_id2 )
			self.disorder_dict[uni_id2] = disorder_positions2

		p1_pos = np.arange( int( start1 ), int( end1 ) + 1, 1 )
		p2_pos = np.arange( int( start2 ), int( end2 ) + 1, 1 )

		# Binary vectors indicating positions corresponding to disordered residues.
		p1_pos = np.array( [1 if pos in disorder_positions1 else 0 for pos in p1_pos] ).reshape( -1, 1 )
		p2_pos = np.array( [1 if pos in disorder_positions2 else 0 for pos in p2_pos] ).reshape( -1, 1 )

		# Pad the binary disorder position vectors.
		p1_pad = np.zeros( [self.max_len, 1] )
		m = p1_pos.shape[0]
		p1_pad[:m, :] = p1_pos
		p1_pos = p1_pad

		p2_pad = np.zeros( [self.max_len, 1] )
		m = p2_pos.shape[0]
		p2_pad[:m, :] = p2_pos
		p2_pos = p2_pad

		print( obj, "  ", cg, " --> ", np.count_nonzero( p1_pos ), "  ", np.count_nonzero( p2_pos ) )
		if "interaction" in obj:
			# IDR-IDR interactions.
			disorder_mat1 = p1_pos*p2_pos.T
			# IDR-any interactions.
			disorder_mat2 = p1_pos+p2_pos.T
			disorder_mat2 = np.where( disorder_mat2 >= 1, 1, 0 )

			# Coarse grain the matrices.
			if int( cg ) > 1:
				m = nn.MaxPool2d( kernel_size = int( cg ), stride = int( cg ) )

				disorder_mat1 = torch.from_numpy( disorder_mat1 ).unsqueeze( 0 ).unsqueeze( 0 ).float()
				disorder_mat1 = m( disorder_mat1 ).squeeze( [0, 1] ).numpy()
				disorder_mat2 = torch.from_numpy( disorder_mat2 ).unsqueeze( 0 ).unsqueeze( 0 ).float()
				disorder_mat2 = m( disorder_mat2 ).squeeze( [0, 1] ).numpy()

			self.logs["counts"][f"{obj}_{cg}"]["IDR-IDR_interactions"] += np.count_nonzero( disorder_mat1 )
			self.logs["counts"][f"{obj}_{cg}"]["IDR-any_interactions"] += np.count_nonzero( disorder_mat2 )

			return disorder_mat1, disorder_mat2

		elif "interface" in obj:
			# Coarse grain the matrices.

			if int( cg ) > 1:
				m = nn.MaxPool1d( kernel_size = int( cg ), stride = int( cg ) )

				p1_pos = torch.from_numpy( p1_pos.T ).unsqueeze( 0 ).float()
				p1_pos = m( p1_pos ).squeeze( 0 ).numpy().T
				p2_pos = torch.from_numpy( p2_pos.T ).unsqueeze( 0 ).float()
				p2_pos = m( p2_pos ).squeeze( 0 ).numpy().T

				print( p1_pos.shape, "  ", p2_pos.shape )
			self.logs["counts"][f"{obj}_{cg}"]["P1_IDR"] += np.count_nonzero( p1_pos )
			self.logs["counts"][f"{obj}_{cg}"]["P2_IDR"] += np.count_nonzero( p2_pos )

			disorder_pos = np.concatenate( [p1_pos, p2_pos], axis = 0 )
			return disorder_pos, None

		else:
			raise Exception( f"Unrecognized objective {cg}..." )



	def predict( self ):
		"""
		Predict cmap for the input protein pair from all models.
		Store all predictions in anested  dict:
			self.predictions{
				header: {
					cg: model output
				}
			}
			header --> identifier for input pair as specified in input file.
			cg --> coarse grainng level.
		In test mode, also plot the distribution of contact counts, 
			prot1/2 lengths for the OOD set.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		
		# For all entries in OOD set.
		for idx, key in enumerate( self.prot1_emb.keys() ):
			head1, head2 = key.split( "--" )
			head2, num = head2.split( "_" )
			uni_id1, _, _ = head1.split( ":" )
			uni_id2, _, _ = head2.split( ":" )
			header = f"{uni_id1}--{uni_id2}_{num}"

			# Ignoring this entry, as AF2-multimer crashed for this.
			if key == "P0DTD1:1743:1808--P0DTD1:1565:1641_1":
				continue

			self.predictions[key] = {}
			prot1_emb = np.array( self.prot1_emb[key] )
			prot2_emb = np.array( self.prot2_emb[key] )
			target_cmap = np.array( self.cmaps[key] )

			# For all objectives (interaction, interface).
			for obj in self.parameters.keys():
				
				# For all coarse grainings.
				for cg in self.parameters[obj].keys():
					model = self.apply_settings( obj, cg )

					prot1, prot2, target, target_mask, eff_len = self.get_input_tensors( key, prot1_emb, prot2_emb, target_cmap )
					
					# get model predictions.
					with torch.no_grad():
						uncal_output = model( prot1, prot2 )
						
						uncal_output = uncal_output*target_mask
					uncal_output, target = self.extract_model_output( uncal_output, target, eff_len )

					_, cg = cg.split( "_" )
					uncal_output = uncal_output.detach().cpu().numpy()

					disorder_mat1, disorder_mat2 = self.get_disorder_matrix( key, obj, cg )
					
					self.predictions[key][f"{obj}_{cg}"] = {
													"Disobind_uncal": uncal_output,
													"masks": target_mask,
													"disorder_mat1": disorder_mat1,
													"disorder_mat2": disorder_mat2,
													"prot1_aa_mask": self.aa_masks[key]["prot1"],
													"prot2_aa_mask": self.aa_masks[key]["prot1"],
													"prot1_slims_mask": self.slims_masks[key]["prot1"],
													"prot2_slims_mask": self.slims_masks[key]["prot1"]
														}

					print( f"{idx} ------------------------------------------------------------\n" )

		# Save the disorder dict on disk.
		if not os.path.exists( self.disorder_file_path ):
			with open( self.disorder_file_path, "w") as w:
				json.dump( self.disorder_dict, w )

		self.plot_ood_dist()


	def extract_model_output( self, output, target, eff_len ):
		"""
		Reshape the model output and target into the required shape 
			i.e. [L1, L2] for interaction.
				[L1+L2, 1] for interface.
		eff_len specifies the effective size post coarse graining.

		Input:
		----------
		output --> (np.array) model predictions.
		target --> (np.array) target cmap.
		eff_len -->  list containing effective lengths for prot1/2.

		Returns:
		----------
		output --> (np.array) model predictions.
		target --> (np.array) target cmap.
		"""
		if "interaction" in self.objective[0]:
			output = output.reshape( eff_len )
			target = target.reshape( eff_len )

		elif "interface" in  self.objective[0]:
			output = output.reshape( ( 2*eff_len[0], 1 ) )
			target = target.reshape( ( 2*eff_len[0], 1 ) )

		return output, target


	def plot_ood_dist( self ):
		"""
		Plot the distribution of contact counts, 
			prot1/2 lengths for the OOD set.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		fig, axis = plt.subplots( 1, 3, figsize = ( 20, 15 ) )
		axis[0].hist( self.counts[0], bins = 20 )
		axis[1].hist( self.counts[1], bins = 20 )
		axis[2].hist( self.counts[2], bins = 20 )
		axis[0].xaxis.set_tick_params( labelsize = 14, length = 8, width = 2 )
		axis[1].xaxis.set_tick_params( labelsize = 14, length = 8, width = 2 )
		axis[2].xaxis.set_tick_params( labelsize = 14, length = 8, width = 2 )
		axis[0].yaxis.set_tick_params( labelsize = 14, length = 8, width = 2 )
		axis[1].yaxis.set_tick_params( labelsize = 14, length = 8, width = 2 )
		axis[2].yaxis.set_tick_params( labelsize = 14, length = 8, width = 2 )
		axis[0].set_ylabel( "Counts", fontsize = 16 )
		axis[1].set_ylabel( "Counts", fontsize = 16 )
		axis[2].set_ylabel( "Counts", fontsize = 16 )
		axis[0].set_title( "OOD set Contacts distribution", fontsize = 16 )
		axis[1].set_title( "OOD set Prot1 lengths distribution", fontsize = 16 )
		axis[2].set_title( "OOD set Prot2 lengths distribution", fontsize = 16 )

		plt.savefig( f"./{self.output_dir}/1_OOD_distribution.png", dpi = 300 )
		plt.close()


#################################################################
import time
tic = time.time()
Prediction().forward()
toc = time.time()
print( f"Time taken = {toc - tic} seconds" )
print( "May the Force be with you..." )
#################################################################

