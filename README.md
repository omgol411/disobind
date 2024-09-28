
[![PubMed](https://salilab.org/imp-systems/static/images/pubmed.png)]()

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10360718.svg)]()

# Disobind
Disobind is a deep learning method for sequence based, partner-dependent contact map and interface prediction.

## Publication and Data
* Kartik Majila, Varun Ullanat, Shruthi Viswanath, __Disobind: __, [DOI]().
* Data is deposited in [Zenodo]()


## Dependencies:
* See `requirements.txt` for Python dependencies


## Installation
1. Clone the repository
```
git clone https://github.com/isblab/disobind.git
```


2. Set up the repository  

Run the following commands in order:
```
cd Disobind/
chmod +x install.sh
./install.sh
```


## Prediction
In order to use Disobind for making predictions, create a csv file as shown in `example/test.csv`.  
Run the following command to use Disobind for the example case with default settings:
```
python run_disobind.py -f ./example/test.csv 
```


## Dataset creation
Move to the `dataset` directory.  

1. Obtain all PDB IDs from databases of disordered proteins including DIBS, MFIB, FuzDB, PBDtot, PDBcdr, DisProt, IDEAL, MobiDB.
```
python 1_disobind_database.py -c 200
```

2. Download all PDB files, obtain relevant info and create binary complexes with the first protein being disordered.
```
python 2_create_database_dataset_files.py -c 200
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
For training the model, move to `/src` directory.  

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
This script creates a dict containing the following outputs for all tasks (contact map/interface) across all CG (1/5/10):
1. Uncalibrated Disobind predictions
2. Calibrated Disobind predictions
3. Binary target masks
4. Binary mask fo IDR-IDR interactions (disorder_mat1)
5. Binary mask fo IDR-any interactions (disorder_mat2)


### AF2/AF3 predictions
Run the following script on the terminal, specify `self.af_model` in the constructor:
```
python get_af_prediction.py
```
Check all the paths in the constructor before running the script.  
This script creates a dict containing the following outputs for all tasks (contact map/interface) across all CG (1/5/10) from AF2/AF3:
1. pLDDT and PAE corrected AF2/3 contact map.  
For all OOD set entries, predicted contact maps are zeroed if the ipTM <= 0.75.  


### Perform analysis
Run the following script on the terminal:
```
python analysis.py
```
Check all the paths in the constructor before running the script.  
This script parses Disobind/AF2/AF3 predicted outputs for all tasks and all CG values. Following outputs are generated:
1. OOD set metrics.
2. Plots for the contact map (top 10) and interfaces (all).
3. OOD set calibration plots and raw data for each task.
4. AF2 vs AF3 confidence plot and raw data.
5. Sparsity vs F1 score plot and raw data.
6. Contact density vs Metric (Recall/ Precision/ F1 score) and raw data.



## Information
__Author(s):__ Kartik Majila, Varun Ullanat, Shruthi Viswanath

__Date__: October , 2024

__License:__ [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License.

__Testable:__ Yes

__Parallelizeable:__ Yes

__Publications:__  Majila K., _et_. al. DOI: []().

