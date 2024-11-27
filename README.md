#TODO folder names: caps and small
#TODO  : separate the predictions from disobind training and dataset creatation
#TODO add how to run with AF2 . Mention that default is without.
#TODO interface 1 is the default
#TODO example with multiple rows, AF.
#TODO have outputs of example.

[![PubMed](https://salilab.org/imp-systems/static/images/pubmed.png)]()

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10360718.svg)]()

# Disobind
Disobind is a deep learning method for predicting inter-protein contact maps and interface residues for an IDR and its binding partner from their sequences. 

## Publication and Data
* Kartik Majila, Varun Ullanat, Shruthi Viswanath, __Disobind: __, [DOI]().
* Data is deposited in [Zenodo]()


## Dependencies:
* See `requirements.txt` for Python dependencies


## Installation
1. Install Conda  
If not already installed, installed Conda as specified here: https://docs.conda.io/projects/conda/en/latest/index.html.


2. Clone the repository
```
git clone https://github.com/isblab/disobind.git
```

3. Set up the repository  

Run the following commands in order:
```
cd disobind/
chmod +x install.sh
./install.sh
```

For using GPUs, ensure CUDA-toolkit (version 11.8) and the NVIDIA drivers are installed on the system.

## Prediction
The input is a CSV file.

Each row corresponds to one sequence fragment pair for which the Disobind prediction is required. 

Each row contains the UniProt ID, start, and end UniProt residue positions for each of the two protein sequence fragments.  

To run a Disobind prediction only, provide the input as:
`UniProt_ID1,start1,end1,UniProt_ID2,start2,end2`.

To run a Disobind+AF2 prediction, provide the input as:
`UniProt_ID1,start1,end1,UniProt_ID2,start2,end2,AF2_struct_file_path,AF2_pkl_file_path`.

As an example see `example/test.csv`.  

Run the following command to use Disobind for the example case with default settings:

```
python run_disobind.py -f ./example/test.csv 
```

By default, Disobind provides interface predictions at a coarse-grained (CG) resolution 10.

### Other options
| Flags  |                                     Description                                                                           |
| ------ | --------------------------------------------------------------------------------------------------------------------------|
| -f     | path to the input csv file.                                                                                               |
| -c     | no. of cores to be used for downloading the UniProt sequences (default = 2).                                              |
| -o     | output directory name (default: `output`).                                                                                |
| -d     | device to be used - cpu/cuda (default: `cpu`).                                                                            |
| -cm    | whether to predict inter-protein contact maps (default: `False`). By default, only interface residues are predicted.      |
| -cg    | coarse-grained resolution - 0, 1, 5, 10 (default: `1`). If `0`, predictions at all resolutions (1,5 and 10) are provided. |


This script outputs the following files:  

* A CSV output file for all predictions for all input sequence fragment pairs.

* `Predictions.npy`: contains predictions for all input sequence fragment pairs in a nested dictionary.

## Dataset creation
Move to the `dataset` directory.  

1. Obtain all PDB IDs from databases of disordered proteins including DIBS, MFIB, FuzDB, PBDtot, PDBcdr, DisProt, IDEAL, MobiDB.
```
python 1_disobind_database.py -c 250
```

2. Download all PDB files, obtain relevant info and create binary complexes with the first protein being disordered.
```
python 2_create_database_dataset_files.py -c 250
```

3. Create merged binary complexes.
```
python 3_create_merged_binary_complexes.py -c 200
```


4. Create training and OOD test set.
```
python 4_create_non_redundant_dataset.py -c 100
```

5. Obtain embeddings for the dataset and split into Train:Dev:Test set.
```
python create_input_embeddings.py
```


## Model training
For training the model, move to `src` directory.  

Specify the configurations in the `model_versions.py` file and run to create a CONFIG_FILE:  
```
python model_versions.py
```

Next, start model training using:
```
python hparams_search.py -f [CONFIG_FILE] -m manual
```


## Analysis
Move to the `analysis` directory.  

Disobind analysis occurs in 3 steps:
1. Get Disobind predictions on the OOD set.
2. Get AF2/AF3 predictions on the OOD set.
3. Perform analysis (calculate metrics on OOD, create calibration plots, etc.) using Disobind and AF2/AF3 predictions.

### Disobind predictions
Run the following script on the terminal:
```
python predict.py
```
Check all the paths in the constructor before running the script.  

This script creates a dictionary containing the following outputs for all tasks (contact map and interface residue prediction, across CG resolutions: 1, 5, 10):

1. Uncalibrated Disobind predictions
2. Calibrated Disobind predictions
3. Binary target masks
4. Binary mask of interactions between disordered residues (IDR-IDR interactions; disorder_mat1) 
5. Binary mask of interactions between disordered residues and any other residues (IDR-any interactions; disorder_mat2)

### AF2/AF3 predictions

Run the following script on the terminal, specify `self.af_model` in the constructor:
```
python get_af_prediction.py
```
Check all the paths in the constructor before running the script.  

The contact maps from AF2/AF3 predicted structures are corrected based on the pLDDT, PAE, and ipTM cutoffs if any. 
The output is a dictionary for all tasks (contact map and interface residue prediction, across CG resolutions: 1, 5, 10) from AF2 and AF3. 

### Perform analysis
Run the following script on the terminal:

```
python analysis.py
```

Check all the paths in the constructor before running the script.  

This script parses Disobind/AF2/AF3 predicted outputs for all tasks and all CG values. Following outputs are generated:
1. OOD set metrics.
2. OOD set calibration plots and raw data for the plots. 
3. AF2 vs AF3 confidence plot and raw data for the plots. 
4. Sparsity vs F1 score plot and raw data for the plots. 
5. Predicted interfaces at CG 1 for case specific analysis in a .txt file.
   

## Information
__Author(s):__ Kartik Majila, Varun Ullanat, Shruthi Viswanath

__Date__: October , 2024

__License:__ [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License.

__Testable:__ Yes

__Parallelizeable:__ Yes

__Publications:__  Majila K., _et_. al. DOI: []().

