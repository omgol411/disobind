# Dataset creation
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

# Comparison to competing methods 

1. Create inputs to alternate methods. 
