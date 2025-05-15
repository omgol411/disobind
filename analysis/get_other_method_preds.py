"""
Obtain predictions from interface predictors - AIUPred and DeepDisoBind.
"""
import os,json
from typing import Dict
import numpy as np
from Bio import SeqIO
import aiupred_lib


class Othermethods():
	"""
	Parse the DeepDisoBind prediction results and arange
		in Disobind output format.
	Get AIUPred predictions for the OOD set
		and arange in Disobind output format.
	"""
	def __init__( self ):
		self.input_dir = "../database/other_methods"
		self.output_dir = "./other_methods/"

		self.max_len = 200

		# Prefix for all DeepDisoBind result files.
		self.deepdiso_preds_prefix = "deepdisobind_fasta"
		# AIUPred input JSON file.
		self.aiupred_input_file = os.path.join( self.input_dir, "aiupred_input.json" )

		self.output_file = os.path.join( self.output_dir, "other_methods.npy" )

		self.predictions = {k:{} for k in ["aiupred", "deepdisobind"]}


	def forward( self ):
		self.create_dir()
		with open( self.aiupred_input_file, "r" ) as f:
			self.aiupred_input = json.load( f )
		aiupred_preds = self.get_aiupred_predictions()
		deepdiso_preds = self.get_deepdisobind_predictions()
		self.assemble_interfaces_for_ood_entries( aiupred_preds, deepdiso_preds )

		np.save( self.output_file, self.predictions, allow_pickle = True )


	def create_dir( self ):
		"""
		Create the required directories if they don't already exist.
		"""
		os.makedirs( self.output_dir, exist_ok = True )


	def load_aiupred_model( self ):
		"""
		Load the embedding regression models.
		"""
		embedding_model, regression_model, device = aiupred_lib.init_models( "binding" )
		return embedding_model, regression_model, device


	def get_aiupred_predictions( self ) -> Dict[str, np.array]:
		"""
		Obtain predictions for all OOD sequences.

		aiupred_lib.low_memory_predict_disorder() throws an error:
			return savgol_filter(transformed_pred, 11, 5)
			NameError: name 'transformed_pred' is not defined
		aiupred_lib.predict_binding() throws the same error.

		Incorrect usage on GitHub for predict_binding().
			Do not mention argument bindin=True (default: False).
			With binding=False, smoothed transformer predictions are returned instead.
			Correct usage:
				predict_binding(sequence, embedding_model, reg_model, device, binding=True, smoothing=True)
					in aiupred_lib.py --> aiupred_binding()
		"""
		aiupred_preds = {}
		print( "\n--> Obtaining AIUPred predictions..." )
		embedding_model, regression_model, device = self.load_aiupred_model()
		for seq_id in self.aiupred_input:
			sequence = self.aiupred_input[seq_id]["seq"]
			prediction = aiupred_lib.predict_binding( sequence,
													embedding_model,
													regression_model,
													device, binding = True, smoothing = True )
			# AIUPred paper uses a threshold of 0.5 to convert the propensity to binary values.
			prediction = np.where( prediction > 0.5, 1, 0 )

			# Arrange all partner-independent predictions according to the
			# 	OOD entry they belong to.
			ood_entry_id = self.aiupred_input[seq_id]["ood_entry_id"]
			aiupred_preds[ood_entry_id] = {seq_id: prediction}
		return aiupred_preds


	def get_deepdisobind_predictions( self ) -> Dict[str, np.array]:
		"""
		Parse the DeepDisoBind predictions form the results FASTA files.
		We need "protein_binary" which represents the binarized protein binding propensity.
		The output being a str, we split at "protein_binary:" and "DNA_propensity".
			The sub-string in the middle is the protein_binary values we need.
		"""
		deepdiso_preds = {}
		print( "\n--> Obtaining DeepDisoBind predictions..." )
		for suffix in ["0-20", "20-40", "40-59"]:
			file_path = os.path.join( self.input_dir, f"result_deepdiso_{suffix}.fasta" )

			with open(file_path, 'r') as f:
				for record in SeqIO.parse( f, "fasta" ):
					seq_id = record.id
					result = str( record.seq )
					protein_binary = result.split(
									"protein_binary:")[1].split( "DNA_propensity" )[0]
					prediction = np.array( list( protein_binary ) )

					ood_entry_id = self.aiupred_input[seq_id]["ood_entry_id"]
					deepdiso_preds[ood_entry_id] = {seq_id: prediction}
		return deepdiso_preds


	def pad_to_max_len( self, pred: np.array ) -> np.array:
		"""
		Add 0-padding to max_len.
		"""
		pad_mask = np.zeros( self.max_len )
		m = pred.shape[0]
		pad_mask[:m] = pred
		return pad_mask


	def assemble_interfaces_for_ood_entries( self, aiupred_preds: Dict, deepdiso_preds: Dict ):
		"""
		Pair up the partner-independent interface predictions for all OOD entries.
		"""
		print( "\n--> Assembling Interface predictions for OOD entries..." )
		for ood_entry_id in aiupred_preds:
			if len( aiupred_preds[ood_entry_id] ) > 2 or len( deepdiso_preds[ood_entry_id] ) > 2:
				raise ValueError( f"Too many fragments for OOD entry: {ood_entry_id}..." )
			
			for seq_id in aiupred_preds[ood_entry_id]:
				ood_uni_id1, ood_uni_id2 = ood_entry_id.split( "--" )
				ood_uni_id2, _ = ood_uni_id2.split( "_" )

				seq_uni_id, _, _ = seq_id.split( "_" )

				deepdiso_seq_id = "_".join( seq_id.split( ":" ) )
				# For homomeric entry.
				if len( aiupred_preds[ood_entry_id] ) == 1:
					aiupred_p1 = self.pad_to_max_len( aiupred_preds[ood_entry_id][seq_id] )
					aiupred_p2 = self.pad_to_max_len( aiupred_preds[ood_entry_id][seq_id] )

					deepdiso_p1 = self.pad_to_max_len( deepdiso_preds[ood_entry_id][seq_id] )
					deepdiso_p2 = self.pad_to_max_len( deepdiso_preds[ood_entry_id][seq_id] )
				# Heteromeric entries.
				else:
					if seq_uni_id == ood_uni_id1:
						aiupred_p1 = self.pad_to_max_len( aiupred_preds[ood_entry_id][seq_id] )
						deepdiso_p1 = self.pad_to_max_len( deepdiso_preds[ood_entry_id][seq_id] )
					elif seq_uni_id == ood_uni_id2:
						aiupred_p2 = self.pad_to_max_len( aiupred_preds[ood_entry_id][seq_id] )
						deepdiso_p2 = self.pad_to_max_len( deepdiso_preds[ood_entry_id][seq_id] )
					else:
						raise ValueError( f"Uniprot ID from AIUPred/DeepDisoBind does not match OOD Uniprot ID..." )
				aiupred_interface = np.hstack( [aiupred_p1, aiupred_p2] ).reshape( -1, 1 )
				deepdiso_interface = np.hstack( [deepdiso_p1, deepdiso_p2] ).reshape( -1, 1 )

				self.predictions["aiupred"][ood_entry_id] = aiupred_interface
				self.predictions["deepdisobind"][ood_entry_id] = deepdiso_interface
				# self.predictions[ood_entry_id] = {"aiupred": aiupred_interface,
				# 									"deepdiso": deepdiso_interface
				# 									}


if __name__ == "__main__":
	Othermethods().forward()

