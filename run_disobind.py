########## Make predictions using pre-trained models. ##########
######## ------>"May the Force serve u well..." <------#########
################################################################

############# One above all #############
##-------------------------------------##
import numpy as np
import pandas as pd
import math
from omegaconf import OmegaConf
import os
import subprocess
import argparse
import json
import time
import joblib
import tqdm
from multiprocessing import Pool
import pickle as pkl
import Bio
from Bio.PDB import PDBParser, MMCIFParser


import torch
from torch import nn

from typing import List, Dict

import dataset
from dataset.from_APIs_with_love import get_uniprot_seq
from dataset.utility import get_contact_map
from dataset.create_input_embeddings import Embeddings
from src.models.get_model import get_model
from src.utils import prepare_input
from analysis.params import parameter_files


"""
Script to obtain Disobind predictions for a user.
Inputs:
	Disobind needs the UniProt IDs for both the proteins and the protein 1 (1st in the pair) must be an IDR.
	The input must be provided in a csv file formatas shown below:
	Uni_ID1,start_res1,end_res1,Uni_ID2,start_res2,end_res2

Outputs:
	Disobind predictions for all tasks and all CG resolutions.
"""

class Disobind():
	def __init__( self, args ):
		"""
		Constructor
		"""
		# Input file containing the prot1/2 headers.
		self.input_file = args.f
		# No. of CPU cores for parallelism.
		self.cores = args.c
		# Predict cmaps.
		self.predict_cmap = args.cm
		# Coarse grained resolutions.
		self.required_cg = args.cg
		# Name for output directory.
		self.output_dir = args.o

		# Embedding type to be used for prediction.
		self.embedding_type = "T5"
		# Use global/local embeddings.
		self.scope = "global"
		# Device to be used for running Disobind.
		self.device = args.d
		# Max protein length.
		self.max_len = 100
		# Contact probability threshold.
		self.threshold = 0.5
		# Distance cutoff for defining a contact.
		self.dist_threshold = 8
		# pLDDT cutoff.
		self.plddt_threshold = 70
		# PAE cutoff.
		self.pae_threshold = 5
		# Batch size for obtaining embeddings.
		self.batch_size = 200
		# Objective settings to be used for prediction.
		self.objective = ["", "", "", ""]
		# Load a dict storing paths for each model.
		self.parameters = parameter_files()
		# Dict to store predictions for all tasks.
		self.predictions = {}

		# Dict to store Uniprot sequences.
		self.uniprot_seq = {}

		# Dir containing the Disobind models.
		self.model_dir = "./models/"
		# Filename to store predictions.
		self.output_filename = "Predictions"

		if os.path.exists( self.output_dir ):
			reply = input( "Output directory already exists. Abort? (Y/n)\t" )
			if reply == "Y":
				exit()
		else:
			os.makedirs( self.output_dir, exist_ok = True )
		
		# Absolute path to the analysis dir.
		self.abs_path = os.getcwd()
		# Path for the FASTA file for prot1/2 sequences.
		self.fasta_file = f"{self.abs_path}/{self.output_dir}/p1_p2_test.fasta"
		# Path fo rthe h5 file for prot1/2 embeddings.
		self.emb_file = f"{self.abs_path}/{self.output_dir}/p1_p2_test.h5"
		# Uniprot seq file.
		self.uni_seq_file = f"{self.abs_path}/{self.output_dir}/UniProt_seq.json"


	def forward( self ):
		"""
		Get the Uniprot sequences for all unique Uniprot IDs amongst prot1/2.
			For test mode,
				Use the downloaded Uniprot sequences.
			For predict mode,
				Download the sequences.
		"""
		headers, af_paths = self.read_csv_input()

		prot_pairs = self.process_input_pairs( headers )

		self.get_predictions( prot_pairs, af_paths )

		np.save( f"./{self.output_dir}/{self.output_filename}.npy", self.predictions )



	def get_required_tasks( self ):
		"""
		Get all tasks required by the user.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		tasks --> list of user required tasks.		
		"""
		tasks = []

		if self.required_cg not in [0, 1, 5, 10]:
			raise Exception( "Incorrect coarse-grained resolution specified. Choose from [0, 1, 5, 10]..." )

		for obj in ["interaction", "interface"]:
			if obj == "interaction" and not self.predict_cmap:
				continue
			
			for cg in [1, 5, 10]:
				if self.required_cg == 0:
					tasks.append( f"{obj}_{cg}" )

				elif cg == self.required_cg:
					tasks.append( f"{obj}_{cg}" )

		print( f"Running Disobind for the followinng tasks: \n{tasks}" )
		return tasks


	def get_predictions( self, prot_pairs, af_paths ):
		"""

		Input:
		----------
		prot_pairs --> dict containig list of fragment pairs for all protein pairs.

		Returns:
		----------
		None
		"""
		total_pairs = len( prot_pairs )

		required_tasks = self.get_required_tasks()


		for start in np.arange( 0, total_pairs, self.batch_size ):
			t_start = time.time()
			if start + self.batch_size >= total_pairs:
				end = total_pairs
			else:
				end = start + self.batch_size
			
			print( f"\n{start}:{end}-----------------------------------------" )
			batch = prot_pairs[start:end]
			
			print( "Creating global embeddings for the input sequences..." )
			self.create_embeddings( batch )

			os.chdir( self.abs_path )

			print( "Running predictions..." )
			batch_preds = self.predict( required_tasks, af_paths )
			self.predictions.update( batch_preds )

			t_end = time.time()
			print( f"Time taken for batch {start}-{end} = {( t_end - t_start )/60} minutes\n" )

			subprocess.call( ["rm", f"{self.emb_file}", f"{self.fasta_file}"] )


###################################################################################
##-------------------------------------------------------------------------------##
###################################################################################
	def read_csv_input( self ):
		"""
		Read from a csv file containing prot1 and prot2 info as:
			Uni_ID1,start1,end1,Uni_ID2,start2,end2
		To combine AF2 and Disobind predictions, provide the path to the AF2 model and .pkl file as:
			Uni_ID1,start1,end1,Uni_ID2,start2,end2,model_file_path,pkl_file_path

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		headers --> list of entry_ids for al binary complexes.
		"""
		headers = []
		af_paths = {}
		with open( self.input_file, "r" ) as f:
			input_pairs = f.readlines()

		# Validate if the input is in the correct format:
		for pair in input_pairs:
			pair = pair.strip()
			# Ignore empty string if present in the input.
			if len( pair.split( "," ) ) == 6:
				uni_id1, start1, end1, uni_id2, start2, end2 = pair.split( "," )
				af_model_file, af_pae_file = None, None

			elif len( pair.split( "," ) ) == 8:
				uni_id1, start1, end1, uni_id2, start2, end2, af_model_file, af_pae_file = pair.split( "," )

			else:
				raise Exception( f"Incorrect input format..." )

			entry_id = f"{uni_id1}:{start1}:{end1}--{uni_id2}:{start2}:{end2}_0"
			headers.append( entry_id )
			af_paths[entry_id] = [af_model_file, af_pae_file]

		return headers, af_paths


	# def get_af_pred( self, model_file, pae_file ):
	# 	"""
	# 	Parse the AF2 predicted structure if provided.
	# 	Obtain contact map, binary pLDDT and PAE matrices.
	# 	Get contact map with only high confidence contacts:
	# 		pLDDT >= 70
	# 		PAE <= 5
	# 	Pad the contact map up to max_len.

	# 	Input:
	# 	----------
	# 	model_file --> PDB file for AF2 predicted structure.
	# 	pae_file --> pkl file containing the PAE matrix for the AF2 prediction.

	# 	Returns:
	# 	----------
	# 	af2_pred --> (np.array) AF2 predicted contact map.
	# 	"""
	# 	with open( pae_file, "rb" ) as f:
	# 		data = pkl.load( f )
	# 	pae = data["predicted_aligned_error"]

	# 	coords_dict = {}
	# 	plddt_dict = {}
	# 	if ".pdb" in model_file:
	# 		models = PDBParser().get_structure( "pdb", model_file )
	# 	elif ".cif" in model_file:
	# 		models = MMCIFParser().get_structure( "cif", model_file )
	# 	else:
	# 		raise Exception( "Incorrect file format for the AF2 prediction..." )

	# 	for model in models:
	# 		for chain in model:
	# 			coords_dict[chain.id[0]] = []
	# 			plddt_dict[chain.id[0]] = []
	# 			for residue in chain:
	# 				# Take only the ATOM entries and ignore the 
	# 				# 		HETATM entries (which contain "w" instead of " ").
	# 				if residue.id[0] == " ":
	# 					coords_dict[chain.id[0]].append( residue['CA'].coord )
	# 					plddt_dict[chain.id[0]].append( residue["CA"].get_bfactor() )
	# 			coords_dict[chain.id[0]] = np.array( coords_dict[chain.id[0]] )

	# 	plddt_mat, pae_mat = self.get_plddt_pae_mat( plddt_dict, pae )
	# 	pae_mat = np.where( pae_mat <= self.pae_threshold, 1, 0 )

	# 	chainA, chainB = coords_dict.keys()
	# 	contact_map = get_contact_map( coords_dict[chainA], coords_dict[chainB], self.dist_threshold )

	# 	m, n = contact_map.shape
	# 	pad = np.zeros( ( self.max_len, self.max_len ) )
	# 	pad[:m, :n] = contact_map
	# 	contact_map = pad

	# 	af2_pred = contact_map*plddt_mat*pae_mat

	# 	return af2_pred



	# def get_plddt_pae_mat( self, plddt_dict, pae ):
	# 	"""
	# 	Create a binary matrix based on:
	# 		plDDT values of prot1/2 for AF2 structures.
	# 			plDDT >= 70 --> 1 else 0
	# 		PAE matrix
	# 			plDDT <= 5 --> 1 else 0

	# 	Input:
	# 	----------
	# 	plddt_dict --> dict containing Ca-plddt for all residues in all chains.
	# 	pae --> AF2/3 predicted PAE matrix.

	# 	Returns:
	# 	----------
	# 	plddt_mat --> binary mask indicating residues with high pLDDT.
	# 	pae_mat --> binary mask indicating residues with high PAE.
	# 	"""
	# 	chain1, chain2 = list( plddt_dict.keys() )
	# 	plddt1 = np.array( plddt_dict[chain1] ).reshape( -1, 1 )
	# 	plddt2 = np.array( plddt_dict[chain2] ).reshape( 1, -1 )
	# 	plddt1 = np.where( plddt1 >= self.plddt_threshold, 1, 0 )
	# 	plddt2 = np.where( plddt2 >= self.plddt_threshold, 1, 0 )
	# 	plddt_mat = plddt1*plddt2

	# 	m, n = plddt_mat.shape
	# 	padded_mat = np.zeros( ( self.max_len, self.max_len ) )

	# 	padded_mat[:m, :n] = plddt_mat
	# 	plddt_mat = padded_mat

	# 	# Taking the upper-right and lower-left quadrants.
	# 	pae_ur = pae[:m, m:]
	# 	pae_ll = pae[m:, :m]
	# 	pae = ( pae_ur + pae_ll.T )/2

	# 	if m != pae.shape[0] or n != pae.shape[1]:
	# 		raise Exception( "Incorrect PAE matrix dimensions..." )

	# 	m, n = pae.shape
	# 	pae_mat = np.zeros( ( self.max_len, self.max_len ) )
	# 	pae_mat[:m, :n] = pae

	# 	# Create a binary PAE matrix to highlight confident predictions.
	# 	pae_mat = np.where( pae_mat <= self.pae_threshold, 1, 0 )

	# 	return plddt_mat, pae_mat


###################################################################################
##-------------------------------------------------------------------------------##
###################################################################################
	def parallelize_uni_seq_download( self, uni_id ):
		"""
		Obtain all unique UniProt IDs in the provided input.
		Download all unique UniProt sequences.

		Input:
		----------
		uni_id --> (str) UniProt ID protein.

		Returns:
		----------
		uni_id --> (str) UniProt ID protein.
		seq --> (str) UniProt seq for protein.
		"""
		seq = get_uniprot_seq( uni_id, max_trials = 10, wait_time = 20, return_id = False )

		return uni_id, seq


	def get_unique_uni_ids( self, headers ):
		"""
		Given a list of all protein pairs, get all the unique Uniprot IDs.
		"""
		unique_uni_ids = []
		for head in headers:
			head1, head2 = head.split( "--" )
			uni_id1 = head1.split( ":" )[0]
			uni_id2 = head2.split( ":" )[0]

			if uni_id1 not in unique_uni_ids:
				unique_uni_ids.append( uni_id1 )
			if uni_id2 not in unique_uni_ids:
				unique_uni_ids.append( uni_id2 )

		return unique_uni_ids



	def download_uniprot_seq( self, headers ):
		"""
		Obtain all unique UniProt IDs in the provided input.
		Download all unique UniProt sequences.

		Input:
		----------
		headers --> list of entry_id for a binary complexes.

		Returns:
		----------
		None
		"""
		unique_uni_ids = self.get_unique_uni_ids( headers )

		with Pool( self.cores ) as p:
			for result in tqdm.tqdm( p.imap_unordered( 
													self.parallelize_uni_seq_download, 
													unique_uni_ids ), 
										total = len( unique_uni_ids ) ):
				uni_id, seq = result

				if len( seq ) != 0:
					self.uniprot_seq[uni_id] = seq
				else:
					raise Exception( f"Unable o download seq for Uniprot ID: {uni_id}. Please retry..." )

		print( "Unique Uniprot IDs = ", len( unique_uni_ids ) )
		print( "Total Uniprot sequences obtained = ", len( self.uniprot_seq ) )


###################################################################################
##-------------------------------------------------------------------------------##
###################################################################################
	def process_input_pairs( self, headers ):
		"""
		For all input pairs, convert them into a uniform format:
			UniID1:start1:end1--UniID2:start2:end2_num

		Input:
		----------
		headers --> list of entry_id for a binary complexes.

		Returns:
		----------
		processed_pairs --> list of protein 1/2 fragment pairs to obtain Disobind predictions for.
		"""
		print( "\nDownloading UniProt sequences..." )
		if not os.path.exists( self.uni_seq_file ):
			self.download_uniprot_seq( headers )

			with open( self.uni_seq_file, "w" ) as w:
				json.dump( self.uniprot_seq, w )
		else:
			with open( self.uni_seq_file, "r" ) as f:
				self.uniprot_seq = json.load( f )

		prot_pairs = []
		for head in headers:
			head1, head2 = head.split( "--" )
			uni_id1, start1, end1 = head1.split( ":" )
			uni_id2, start2, end2 = head2.split( ":" )

			if uni_id1 not in self.uniprot_seq.keys() or uni_id2 not in self.uniprot_seq.keys():
				print( f"Skipping the input pair -- {head}" )		

			else:
				prot_pairs.append( head )
		
		return prot_pairs



###################################################################################
##-------------------------------------------------------------------------------##
###################################################################################
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
		model_config = f"./params/Model_config_{model_ver}.yml"
		model_config = OmegaConf.load( model_config )
		model_config = model_config.conf.model_params

		model_ver = model_ver.split( "_" )
		# Model name.
		mod = "_".join( model_ver[:-1] )
		# model version.
		ver = model_ver[-1]
		# Load Disobind model.
		model = get_model( model_config ).to( self.device )
		model.load_state_dict( 
							torch.load( f"{self.model_dir}{mod}/Version_{ver}/{params_file}.pth", 
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
		key2 --> specifies the coarse graining to be used.

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



	def get_input_tensors( self, prot1, prot2 ):
		"""
		Prepare prot1, prot2, target tensors according to the objective.
			If self.pad is True:
				Pad the prot1/2 and cmap arrys.
				Create a binary target_mask.
		Convert the input arrays to torch.tensor and load on the specified device.
		**Note: If not using padding, terminal residues might be lost upon coarse graining 
				if protein length not divisible by bin_size.

		Input:
		----------
		prot1 --> (np.array) prot1 embedding of dimension [L1, C].
		prot1 --> (np.array) prot2 embedding of dimension [L2, C].

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
		target = target_padded

		target = torch.from_numpy( target )
		target_mask = torch.from_numpy( target_mask )

		prot1 = torch.from_numpy( prot1 ).unsqueeze( 0 )
		prot2 = torch.from_numpy( prot2 ).unsqueeze( 0 )

		prot1, prot2, target, target_mask = prepare_input( prot1, prot2, None, 
															[True, target_mask], 
															objective = self.objective[0], 
															bin_size = self.objective[1], 
															bin_input = self.objective[2], 
															single_output = self.objective[3] )
		prot1 = prot1.to( self.device ).float()
		prot2 = prot2.to( self.device ).float()
		target_mask = target_mask.to( self.device )

		return prot1, prot2, target, target_mask, eff_len



	def predict( self, required_tasks, af_paths ):
		"""
		Predict cmap for the input protein pair from all models.
		Store all predictions in a nested  dict:
			self.predictions{
				pair_id: {
						entry_id: {
							{obj}_{cg}: {
									"Disobind_uncal",
									"Final preds",
							}
						}
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
		predictions = {}
		
		# For all entries in OOD set.
		for idx, entry_id in enumerate( self.prot1_emb.keys() ):
			head1, head2 = entry_id.split( "--" )
			head2, num = head2.split( "_" )
			uni_id1, start1, end1 = head1.split( ":" )
			uni_id2, start2, end2 = head2.split( ":" )
			# header = f"{uni_id1}--{uni_id2}_{num}"

			# Pair does not have the _0 at the end.
			pair_id = f"{uni_id1}::--{uni_id2}::"
			if pair_id not in predictions.keys():
				predictions[pair_id] = {}

			predictions[pair_id][entry_id] = {}
			
			prot1_emb = np.array( self.prot1_emb[entry_id] )
			prot2_emb = np.array( self.prot2_emb[entry_id] )

			# For all objectives (interaction, interface).
			for obj in self.parameters.keys():
				
				# For all coarse grainings.
				for cg in self.parameters[obj].keys():
					if f"{obj}_{cg.split( '_' )[1]}" not in required_tasks:
						continue
					
					model = self.apply_settings( obj, cg )

					prot1, prot2, target, target_mask, eff_len = self.get_input_tensors( 
																						prot1 = prot1_emb, 
																						prot2 = prot2_emb )
					
					# get model predictions.
					with torch.no_grad():
						uncal_output = model( prot1, prot2 )
						
						uncal_output = uncal_output*target_mask
					
					uncal_output = uncal_output.detach().cpu().numpy()
					target_mask = target_mask.detach().cpu().numpy()

					if "interaction" in self.objective[0]:
						uncal_output = uncal_output.reshape( eff_len )
					elif "interface" in self.objective[0]:
						uncal_output = uncal_output.reshape( 2*eff_len[0], 1 )

					_, cg = cg.split( "_" )

					# Get AF2 pred for the entry.
					model_file, pae_file = af_paths[entry_id]
					# If AF2 input is not provided.
					if model_file != None:
						af_obj = AfPrediction( struct_file_path = model_file, data_file_path = pae_file )
						af2_pred = af_obj.get_confident_interactions( [int( start1 ), int( end1 )],
																	[int( start2 ), int( end2 )] )

						# af2_pred = self.get_af_pred( model_file = model_file, pae_file = pae_file )
						af2_pred = self.process_af2_pred( af2_pred )

						# Get Disobind+AF2 output.
						m, n = uncal_output.shape
						af2_diso = np.stack( [uncal_output.reshape( -1 ), af2_pred.reshape( -1 )], axis = 1 )
						output = np.max( af2_diso, axis = 1 ).reshape( m, n )
					else:
						output = uncal_output

					output, df = self.extract_model_output( entry_id, output, eff_len )
					
					predictions[pair_id][entry_id][f"{obj}_{cg}"] = {
																	"Raw_Preds": np.float32( output ),
																	"Final_preds": df
																		}
					print( f"{idx} ------------------------------------------------------------\n" )
		return predictions


	def process_af2_pred( self, af2_pred ):
		"""
		Get contact map or interface residues from AF2 
			predicted contact map for the required CG.

		Input:
		----------
		af2_pred --> (np.array) AF2 predicted contact map.

		Returns:
		----------
		"""
		if self.objective[1] > 1:
			m = nn.MaxPool2d( kernel_size = self.objective[1], stride = self.objective[1] )
			af2_pred = m( torch.from_numpy( af2_pred ).unsqueeze( 0 ).unsqueeze( 0 ) )
			af2_pred = af2_pred.squeeze( [0, 1] ).numpy()

		if "interface" in self.objective[0]:
			p1 = np.zeros( ( af2_pred.shape[0], 1 ) )
			p2 = np.zeros( ( af2_pred.shape[0], 1 ) )

			idx = np.where( af2_pred )
			p1[idx[0]] = 1
			p2[idx[1]] = 1
			af2_pred = np.concatenate( [p1, p2], axis = 1 )

		return af2_pred


	def get_beads( self, cg, prot1, prot2 ):
		"""
		Get the prot1/2 beads corresponding to a residue (cg 1) 
			or a set of residues (cg 5/10).

		Input:
		----------
		cg --> (int) resolution of coarse-graining.
		prot1 --> (str) identifier comprising "uni_id1:start1:end1"
				start1, end1 are the UniProt positions.
		prot2 --> (str) identifier comprising "uni_id2:start2:end2"
				start2, end2 are the UniProt positions.

		Returns:
		----------
		beads1 --> list of baeds at the apprpriate resolution for prot1.
		beads2 --> list of baeds at the apprpriate resolution for prot2.
		"""
		uni_id1, start1, end1 = prot1.split( ":" )
		uni_id2, start2, end2 = prot2.split( ":" )
		start1, end1 = int( start1 ), int( end1 )
		start2, end2 = int( start2 ), int( end2 )

		len_p1 = end1 - start1 + 1
		len_p2 = end2 - start2 + 1
		
		if cg == 1:
			beads1 = np.arange( start1, end1 + 1, 1 )
			beads1 = np.array( list( map( str, beads1 ) ) )
			beads2 = np.arange( start2, end2 + 1, 1 ) 
			beads2 = np.array( list( map( str, beads2 ) ) )
		
		else:
			beads1, beads2 = [], []
			for s in np.arange( start1, end1 + 1, cg ):
				e = s + cg - 1 if s + cg - 1 < end1 else end1
				beads1.append( f"{s}-{e}" )
			beads1 = np.array( beads1 )
			
			for s in np.arange( start2, end2 + 1, cg ):
				e = s + cg - 1 if s + cg - 1 < end2 else end2
				beads2.append( f"{s}-{e}" )
			beads2 = np.array( beads2 )

			# H_out = ( ( H_in + 2*padding - dilation( kernel_size - 1 ) )/stride ) + 1
			# For us padding = 0 and dilation = 1.
			if len_p1 == 100:
				len_p1 = ( ( len_p1 - ( cg - 1 ) )//cg ) + 1
			else:
				len_p1 = math.ceil( ( ( len_p1 - ( cg - 1 ) )/cg ) + 1 )
			
			if len_p2 == 100:
				len_p2 = ( ( len_p2 - ( cg - 1 ) )//cg ) + 1
			else:
				len_p2 = math.ceil( ( ( len_p2 - ( cg - 1 ) )/cg ) + 1 )

		return len_p1, beads1, len_p2, beads2



	def extract_model_output( self, entry_id, output, eff_len ):
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
		obj = self.objective[0]
		cg = self.objective[1]
		prot1, prot2 = entry_id.split( "--" )
		prot2, _ = prot2.split( "_" )

		len_p1, beads1, len_p2, beads2 = self.get_beads( cg, prot1, prot2 )
		
		if "interaction" in obj:
			# output = output.reshape( eff_len )
			output = output[:len_p1, :len_p2]
			idx = np.where( output >= self.threshold )
			df = pd.DataFrame()
			if len( idx[0] ) != 0:
				df["Protein1"] = [prot1]*len( idx[0] )
				df["Residue1"] = beads1[idx[0]]
			else:
				df["Protein1"] = []
				df["Residue1"] = []
			if len( idx[1] ) != 0:
				df["Protein2"] = [prot2]*len( idx[1] )
				df["Residue2"] = beads2[idx[1]]
			else:
				df["Protein2"] = []
				df["Residue2"] = []

		elif "interface" in  obj:
			# output = output.reshape( ( 2*eff_len[0], 1 ) )
			interface1 = output[:eff_len[0]][:len_p1]
			interface2 = output[eff_len[0]:][:len_p2]
			idx1 = np.where( interface1 >= self.threshold )[0]
			idx2 = np.where( interface2 >= self.threshold )[0]

			interface1_beads = beads1[idx1] if len( idx1 ) != 0 else []
			interface2_beads = beads2[idx2] if len( idx2 ) != 0 else []

			if len( interface1_beads ) > len( interface2_beads ):
				for _ in range( len( interface1_beads ) - len( interface2_beads ) ):
					interface2_beads = np.append( interface2_beads, "" )
			else:
				for _ in range( len( interface2_beads ) - len( interface1_beads ) ):
					interface1_beads = np.append( interface1_beads, "" )

			df = pd.DataFrame()
			df["Protein1"] = interface1_beads
			df["Protein2"] = interface2_beads
			output = np.concatenate( ( interface1, interface2 ), axis = 0 )
		df.to_csv( f"{self.abs_path}/{self.output_dir}/{entry_id}_{obj}_cg{cg}.csv" )

		return output, df



###################################################################################
##-------------------------------------------------------------------------------##
###################################################################################
class AfPrediction():
	def __init__( self, struct_file_path: str, data_file_path: str ):
		# AF2/3 structure file path.
		self.struct_file_path = struct_file_path
		# AF2/3 structure data file path.
		self.data_file_path = data_file_path

		# Max length for pad.
		self.max_len = 100
		# Distance threshold in (Angstorm) to define a contact between residue pairs.
		self.dist_threshold = 8
		# pLDDt cutoff to consider a confident prediction.
		self.plddt_cutoff = 70
		# PAE cutoff to consider a confident prediction.
		self.pae_cutoff = 5

		# Biopython Structure object.
		self.structure = self.get_structure( 
									self.get_parser()
									 )
		# Get chains and sanity check for binary complex.
		self.get_chains()
		# Residue positions of all residues for each chain.
		self.get_residue_positions()
		self.get_chain_lengths( self.res_dict )
		# Ca-coords of all residues for each chain.
		self.get_ca_coordinates()
		# Ca-plddt of all residues for each chain.
		self.get_ca_plddt()
		# Average PAE matrix.
		self.get_pae()



	def get_parser( self ):
		"""
		Get the required parser (PDB/CIF) for the input file.
		"""
		ext = os.path.splitext( self.struct_file_path )[1]

		if "pdb" in ext:
			parser = PDBParser()
		elif "cif" in ext:
			parser = MMCIFParser()
		else:
			raise Exception( "Incorrect file format.. Suported .pdb/.cif only." )

		return parser


	def get_chains( self ):
		"""
		Get the total no. of chains in the prediction.
		Only binary complexes are allowed.
		"""
		self.chain_ids = [chain.id for model in self.structure for chain in model]

		if len( self.chain_ids ) > 2:
			raise Exception( "Too many chains. Does support non-binary complexes..." )


	def get_structure( self, parser: Bio.PDB.PDBParser ):
		"""
		Return the Biopython Structure object for the input file.
		"""
		basename = os.path.basename( self.struct_file_path )
		structure = parser.get_structure( basename, self.struct_file_path )

		return structure


	def get_residues( self ):
		"""
		Get all residues in the structure.
		"""
		coords = []
		for model in self.structure:
			for chain in model:
				chain_id = chain.id[0]
				for residue in chain:
					yield residue, chain_id


	def extract_perresidue_quantity( self, residue, quantity: str ): 
		"""
		Given the Biopython residue object, return the specified quantity:
			1. residue position
			2. Ca-coordinate
			3. Ca-pLDDT
		"""
		# Use Ca-atom for all other amino acids.
		rep_atom = "CA"

		if quantity == "res_pos":
			return residue.id[1]

		elif quantity == "coords":
			coords = residue[rep_atom].coord
			return coords
		
		elif quantity == "plddt":
			plddt = residue[rep_atom].bfactor
			return plddt
		
		else:
			raise Exception( f"Specified quantity: {quantity} does not exist..." )


	def get_residue_positions( self ):
		"""
		Get the residue positions for all residues.
		"""
		res_dict = {}
		for residue, chain_id in self.get_residues():
			res_id = self.extract_perresidue_quantity( residue, "res_pos" )
			if chain_id not in res_dict.keys():
				res_dict[chain_id] = np.array( res_id )
			else:
				res_dict[chain_id] = np.append( res_dict[chain_id], res_id )

		self.res_dict = {k: v.reshape( -1, 1 ) for k, v in res_dict.items()}



	def get_chain_lengths( self, res_dict: Dict ):
		"""
		Create a dict containing the length of all chains in the system 
			and the total length of the system.
		"""
		lengths_dict = {}
		lengths_dict["total"] = 0
		for chain in res_dict:
			chain_length = len( res_dict[chain] )
			lengths_dict[chain] = chain_length
			lengths_dict["total"] += chain_length

		self.lengths_dict = lengths_dict


	def get_ca_coordinates( self ):
		"""
		Get the coordinates for all Ca atoms of all residues.
		"""
		coords_dict = {}
		for residue, chain_id in self.get_residues():
			coords = self.extract_perresidue_quantity( residue, "coords" )
			if chain_id not in coords_dict.keys():
				coords_dict[chain_id] = np.array( coords )
			else:
				coords_dict[chain_id] = np.append( coords_dict[chain_id], coords )

		self.coords_dict = {k: v.reshape( -1, 3 ) for k, v in coords_dict.items()}



	def get_ca_plddt( self ):
		"""
		Get the pLDDT score for all Ca atoms of all residues.
		"""
		plddt_dict = {}
		for residue, chain_id in self.get_residues():
			plddt = self.extract_perresidue_quantity( residue, "plddt" )
			if chain_id not in plddt_dict.keys():
				plddt_dict[chain_id] = np.array( [plddt] )
			else:
				plddt_dict[chain_id] = np.append( plddt_dict[chain_id], plddt )

		self.plddt_dict = {k: v.reshape( -1, 1 ) for k, v in plddt_dict.items()}



	def get_data_dict( self ):
		"""
		Parse the AF2/3 data file.
			AF2 data file is saved as a .pkl file 
				whereas for AF3 it's stored as .json.
		"""
		ext = os.path.splitext( self.data_file_path )[1]

		if "json" in ext:
			with open( self.data_file_path, "rb" ) as f:
				data = json.load( f )
		else:
			raise Exception( "Incorrect file format.. Suported .json only." )

		return data[0]



	def get_pae( self ):
		"""
		Return the PAE matrix from the data dict.
			AF2/3 PAE matrix is asymmetric.
			Hence, we consider the average PAE: ( PAE + PAE.T )/2.
		"""
		data = self.get_data_dict()
		# For AF2.
		if "predicted_aligned_error" in data.keys():
			pae = np.array( data["predicted_aligned_error"] )
		else:
			raise Exception( "PAE matrix not found..." )

		self.pae = ( pae + pae.T )/2



	def get_chains_n_indices( self, interacting_region: Dict ):
		"""
		Obtain the chain IDs and residues indices 
			for the required interacting region.
		residue_index = residue_position - 1
		"""
		chain1, chain2 = interacting_region.keys()
		mol1_res1, mol1_res2 = interacting_region[chain1]
		mol1_res1 -= 1
		mol2_res1, mol2_res2 = interacting_region[chain2]
		mol2_res1 -= 1

		return [chain1, chain2], [mol1_res1, mol1_res2], [mol2_res1, mol2_res2]



	def get_required_coords( self, chains: List, mol1_res: List, mol2_res: List ):
		"""
		Get the coordinates for the interacting region 
			for which confident interactions are required.
		"""
		chain1, chain2 = chains
		start1, end1 = mol1_res
		start2, end2 = mol2_res
		coords1 = self.coords_dict[chain1][start1:end1,:]
		coords2 = self.coords_dict[chain2][start2:end2,:]

		return coords1, coords2



	def get_required_plddt( self, chains: List, mol1_res: List, mol2_res: List ):
		"""
		Get the plddt for the interacting region 
			for which confident interactions are required.
		"""
		chain1, chain2 = chains
		start1, end1 = mol1_res
		start2, end2 = mol2_res
		plddt1 = self.plddt_dict[chain1][start1:end1]
		plddt2 = self.plddt_dict[chain2][start2:end2]

		return plddt1, plddt2



	def get_required_pae( self, chains: List, mol1_res: List, mol2_res: List ):
		"""
		Get the PAE matrix for the interacting region.
			For this we need the cumulative residue index 
				uptil the required residue position.
		"""
		chain1, chain2 = chains
		start1, end1 = mol1_res
		start2, end2 = mol2_res

		# Count total residues till start1 and start2.
		cum_start1, cum_start2 = 0, 0
		for chain in self.res_dict:
			if chain == chain1:
				cum_start1 += start1
				break
			else:
				cum_start1 += len( self.res_dict[chain] )

		for chain in self.res_dict:
			if chain == chain2:
				cum_start2 += start2
				break
			else:
				cum_start2 += len( self.res_dict[chain] )

		cum_end1 = cum_start1 + ( end1 - start1 )
		cum_end2 = cum_start2 + ( end2 - start2 )

		pae = self.pae[cum_start1:cum_end1, cum_start2:cum_end2]

		return pae


	def get_interaction_data( self, interacting_region ):
		"""
		Get the interaction amp, pLDDT, and PAE for the interacting region.
		"""
		chains, mol1_res, mol2_res = self.get_chains_n_indices( interacting_region )

		coords1, coords2 = self.get_required_coords( chains, mol1_res, mol2_res )

		# Create a contact map or distance map as specified.
		contact_map = get_contact_map( coords1, coords2, self.dist_threshold )
		# interaction_map = get_interaction_map( coords1, coords2, 
		# 										self.contact_threshold )

		plddt1, plddt2 = self.get_required_plddt( chains, mol1_res, mol2_res )
		pae = self.get_required_pae( chains, mol1_res, mol2_res )

		return contact_map, plddt1, plddt2, pae


	def apply_confidence_cutoffs( self, plddt1, plddt2, pae ):
		"""
		mask low-confidence interactions.
		"""
		plddt1 = np.where( plddt1 >= self.plddt_cutoff, 1, 0 )
		plddt2 = np.where( plddt2 >= self.plddt_cutoff, 1, 0 )
		plddt_matrix = plddt1 * plddt2.T

		pae = np.where( pae <= self.pae_cutoff, 1, 0 )

		return plddt_matrix, pae


	def get_confident_interactions( self, prot1_res, prot2_res ):
		"""
		For the specified regions in the predicted structure, 
			obtain all confident interacting residue pairs.
		Assuming that chain 1 and 2 will correspond to protein 1 and 2 respectively.
		"""
		chain1, chain2 = self.chain_ids
		interacting_region = {}
		interacting_region[chain1] = prot1_res
		interacting_region[chain2] = prot2_res
		contact_map, plddt1, plddt2, pae = self.get_interaction_data( interacting_region )
		plddt_matrix, pae = self.apply_confidence_cutoffs( plddt1, plddt2, pae )
		confident_interactions = contact_map * plddt_matrix * pae

		m, n = confident_interactions.shape
		pad = np.zeros( ( self.max_len, self.max_len ) )
		pad[:m, :n] = confident_interactions
		confident_interactions = pad

		return confident_interactions


#################################################################
if __name__ == "__main__":
	tic = time.time()
	parser = argparse.ArgumentParser( description = "Script to obtain Disobind predictions for the specified protein pairs.")

	parser.add_argument( 
						"--input", "-f", dest = "f", 
						help = "Provide protein pairs as -- Uni_ID1,start_res1,end_res1,Uni_ID2,start_res2,end_res2 -- in a csv file format.", 
						required = True )
	
	parser.add_argument( 
						"--max_cores", "-c", dest = "c", 
						help = "No. of cores to be used.", 
						type = int, required = False, default = 2 )

	parser.add_argument( 
						"--output_dir", "-o", dest = "o", 
						help = "Name of the output directory.", 
						type = str, required = False, default = "output" )

	parser.add_argument( 
						"--device", "-d", dest = "d", 
						help = "Device to be used (cpu/cuda.", 
						type = str, required = False, default = "cpu" )

	parser.add_argument( 
						"--cmaps", "-cm", dest = "cm", 
						help = "If this flag is provided, will predict contact maps.", 
						action = "store_true", required = False, default = False )

	parser.add_argument( 
						"--coarse", "-cg", dest = "cg", 
						help = "Provid the resolution of coarse graining. If 0, will run for all CG.", 
						type = int, required = False, default = 1 )


	args = parser.parse_args()

	Disobind( args ).forward()
	toc = time.time()
	print( f"Time taken = {toc - tic} seconds" )
	print( "May the Force be with you..." )
#################################################################


