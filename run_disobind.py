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
from Bio.PDB import PDBParser, MMCIFParser

import torch
from torch import nn

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
	The input must be provided in a csv file formatas shown beloq:
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
		# base_path = f"/data2/kartik/Disorder_Proteins/Archive/Database_17Apr24/v_{self.version}/"
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


	def get_af_pred( self, model_file, pae_file ):
		"""
		Parse the AF2 predicted structure if provided.
		Obtain contact map, binary pLDDT and PAE matrices.
		Get contact map with only high confidence contacts:
			pLDDT >= 70
			PAE <= 5
		Pad the contact map up to max_len.

		Input:
		----------
		model_file --> PDB file for AF2 predicted structure.
		pae_file --> pkl file containing the PAE matrix for the AF2 prediction.

		Returns:
		----------
		af2_pred --> (np.array) AF2 predicted contact map.
		"""
		with open( pae_file, "rb" ) as f:
			data = pkl.load( f )
		pae = data["predicted_aligned_error"]

		coords_dict = {}
		plddt_dict = {}
		if ".pdb" in model_file:
			models = PDBParser().get_structure( "pdb", model_file )
		elif ".cif" in model_file:
			models = MMCIFParser().get_structure( "cif", model_file )
		else:
			raise Exception( "Incorrect file format for the AF2 prediction..." )

		for model in models:
			for chain in model:
				coords_dict[chain.id[0]] = []
				plddt_dict[chain.id[0]] = []
				for residue in chain:
					# Take only the ATOM entries and ignore the 
					# 		HETATM entries (which contain "w" instead of " ").
					if residue.id[0] == " ":
						coords_dict[chain.id[0]].append( residue['CA'].coord )
						plddt_dict[chain.id[0]].append( residue["CA"].get_bfactor() )
				coords_dict[chain.id[0]] = np.array( coords_dict[chain.id[0]] )

		plddt_mat, pae_mat = self.get_plddt_pae_mat( plddt_dict, pae )
		pae_mat = np.where( pae_mat <= self.pae_threshold, 1, 0 )

		chainA, chainB = coords_dict.keys()
		contact_map = get_contact_map( coords_dict[chainA], coords_dict[chainB], self.dist_threshold )

		m, n = contact_map.shape
		pad = np.zeros( ( self.max_len, self.max_len ) )
		pad[:m, :n] = contact_map
		contact_map = pad

		af2_pred = contact_map*plddt_mat*pae_mat

		return af2_pred



	def get_plddt_pae_mat( self, plddt_dict, pae ):
		"""
		Create a binary matrix based on:
			plDDT values of prot1/2 for AF2 structures.
				plDDT >= 70 --> 1 else 0
			PAE matrix
				plDDT <= 5 --> 1 else 0

		Input:
		----------
		plddt_dict --> dict containing Ca-plddt for all residues in all chains.
		pae --> AF2/3 predicted PAE matrix.

		Returns:
		----------
		plddt_mat --> binary mask indicating residues with high pLDDT.
		pae_mat --> binary mask indicating residues with high PAE.
		"""
		chain1, chain2 = list( plddt_dict.keys() )
		plddt1 = np.array( plddt_dict[chain1] ).reshape( -1, 1 )
		plddt2 = np.array( plddt_dict[chain2] ).reshape( 1, -1 )
		plddt1 = np.where( plddt1 >= self.plddt_threshold, 1, 0 )
		plddt2 = np.where( plddt2 >= self.plddt_threshold, 1, 0 )
		plddt_mat = plddt1*plddt2

		m, n = plddt_mat.shape
		padded_mat = np.zeros( ( self.max_len, self.max_len ) )

		padded_mat[:m, :n] = plddt_mat
		plddt_mat = padded_mat

		# Taking the upper-right and lower-left quadrants.
		pae_ur = pae[:m, m:]
		pae_ll = pae[m:, :m]
		pae = ( pae_ur + pae_ll.T )/2

		if m != pae.shape[0] or n != pae.shape[1]:
			raise Exception( "Incorrect PAE matrix dimensions..." )

		m, n = pae.shape
		pae_mat = np.zeros( ( self.max_len, self.max_len ) )
		pae_mat[:m, :n] = pae

		# Create a binary PAE matrix to highlight confident predictions.
		pae_mat = np.where( pae_mat <= self.pae_threshold, 1, 0 )

		return plddt_mat, pae_mat


###################################################################################
##-------------------------------------------------------------------------------##
###################################################################################
	def parallelize_uni_seq_download( self, head ):
		"""
		Obtain all unique UniProt IDs in the provided input.
		Download all unique UniProt sequences.

		Input:
		----------
		head --> identifier for a binary complexes.

		Returns:
		----------
		uni_id1 --> (str) UniProt ID protein 1.
		seq1 --> (str) UniProt seq for protein 1.
		uni_id2 --> (str) UniProt ID protein 2.
		seq2 --> (str) UniProt seq for protein 2.
		"""
		head1, head2 = head.split( "--" )
		uni_id1, _, _ = head1.split( ":" )
		uni_id2, _, _ = head2.split( ":" )

		if uni_id1 not in self.uniprot_seq.keys():
			seq1 = get_uniprot_seq( uni_id1, max_trials = 10, wait_time = 20, return_id = False )
		else:
			seq1 = self.uniprot_seq[uni_id1]

		if uni_id2 not in self.uniprot_seq.keys() or uni_id1 != uni_id2:
			seq2 = get_uniprot_seq( uni_id2, max_trials = 10, wait_time = 20, return_id = False )
		else:
			seq2 = self.uniprot_seq[uni_id2]

		return uni_id1, seq1, uni_id2, seq2



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
		all_unique_uni_ids = []
		with Pool( self.cores ) as p:
			for result in tqdm.tqdm( p.imap_unordered( self.parallelize_uni_seq_download, headers ), total = len( headers ) ):
				uni_id1, seq1, uni_id2, seq2 = result

				if uni_id1 not in all_unique_uni_ids:
					all_unique_uni_ids.append( uni_id1 )
				if uni_id2 not in all_unique_uni_ids:
					all_unique_uni_ids.append( uni_id2 )

				if len( seq1 ) != 0:
					self.uniprot_seq[uni_id1] = seq1
				if len( seq2 ) != 0:
					self.uniprot_seq[uni_id2] = seq2
		print( "Unique Uniprot IDs = ", len( all_unique_uni_ids ) )
		print( "Total Uniprot sequences obtained = ", len( self.uniprot_seq ) )



###################################################################################
##-------------------------------------------------------------------------------##
###################################################################################
	def process_input_pairs( self, headers ):
		"""
		For all input pairs, convert them into a uniform format:
			UniID1:start1:end1--UniID2:start2:end2_num
		If residue positions for protein 1/2 are not provided, 
				break the sequence into non-overlapping 100 residue long fragments.
			For protein 1, select only IDR containing fragments identified based on annotations from DisProt/ IDEAL/ MobiDB.
			For protein 2, select all fragments.

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

		# Load calibration model.
		cal_model = joblib.load( f"{self.model_dir}{mod}/Version_{ver}/{params_file}.pkl" )

		return model, cal_model



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
		model, cal_model = self.load_model( *self.parameters[obj][cg] )

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
		return model, cal_model



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
									"Disobind_cal",
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
			uni_id1, _, _ = head1.split( ":" )
			uni_id2, _, _ = head2.split( ":" )
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
					
					model, cal_model = self.apply_settings( obj, cg )

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

					# get calibrated model predictions.
					n, h = uncal_output.shape
					cal_output = cal_model.predict( uncal_output.flatten() )
					cal_output = cal_output.reshape( n, h )
					target_mask = target_mask.reshape( n, h )
					cal_output = cal_output*target_mask

					# Get AF2 pred for the entry.
					model_file, pae_file = af_paths[entry_id]
					# If AF2 input is not provided.
					if model_file != None:
						af2_pred = self.get_af_pred( model_file = model_file, pae_file = pae_file )
						af2_pred = self.process_af2_pred( af2_pred )

						# Get Disobind+AF2 output.
						m, n = cal_output.shape
						af2_diso = np.stack( [cal_output.reshape( -1 ), af2_pred.reshape( -1 )], axis = 1 )
						output = np.max( af2_diso, axis = 1 ).reshape( m, n )
					else:
						output = cal_output

					output, df = self.extract_model_output( entry_id, output, eff_len )
					
					predictions[pair_id][entry_id][f"{obj}_{cg}"] = {
																	"Disobind_cal": np.float32( cal_output ),
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
			df["Protein1"] = prot1
			df["Residue1"] = beads1[idx[0]] if len( idx[0] ) != 0 else []
			df["Protein2"] = prot2
			df["Residue2"] = beads2[idx[1]] if len( idx[1] ) != 0 else []

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


