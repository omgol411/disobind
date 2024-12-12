# Analysis
Move to the `analysis` directory.  

Disobind analysis occurs in 3 steps:
1. Get Disobind predictions on the OOD set.
2. Get AF2/AF3 predictions on the OOD set.
3. Perform analysis (calculate metrics on OOD, create calibration plots, etc.) using Disobind and AF2/AF3 predictions.

## Disobind predictions
Run the following script on the terminal:
```
python predict.py
```
Check all the paths in the constructor before running the script.  

This script creates a dictionary containing the following outputs for all tasks (contact map and interface residue prediction, across CG resolutions: 1, 5, 10):

1. Disobind predictions
2. Binary target masks
3. Binary mask of interactions between disordered residues (IDR-IDR interactions; disorder_mat1) 
4. Binary mask of interactions between disordered residues and any other residues (IDR-any interactions; disorder_mat2)

## AF2/AF3 predictions

Run the following script on the terminal, specify `self.af_model` in the constructor:
```
python get_af_prediction.py
```
Check all the paths in the constructor before running the script.  

The contact maps from AF2/AF3 predicted structures are corrected based on the pLDDT, PAE, and ipTM cutoffs if any. 
The output is a dictionary for all tasks (contact map and interface residue prediction, across CG resolutions: 1, 5, 10) from AF2 and AF3. 

## Perform analysis
Run the following script on the terminal:

```
python analysis.py
```

Check all the paths in the constructor before running the script.  

This script parses Disobind/AF2/AF3 predicted outputs for all tasks and all CG values. Following outputs are generated:
1. OOD set metrics in a CSV file format.
2. AF2 vs AF3 confidence plot and raw data for the plots. 
3. Sparsity vs F1 score plot and raw data for the plots. 
4. Predicted interfaces at CG 1 for case specific analysis in a .txt file.
   

