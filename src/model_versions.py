############### Defining configurations for the model ###############
########## ------>"May the Force serve u well..." <------############
#####################################################################

############# One above all #############
##-------------------------------------##
from omegaconf import OmegaConf
import subprocess

proc = subprocess.Popen( "hostname", shell = True, stdout = subprocess.PIPE, )
system = proc.communicate()[0]

"""
Use the following script to define the model configurations for model training.
    The script generates a version_*.yml file used as input for hparams_search.py script.
"""
# Version of the dataset used for developing the model.
dataset_version = 19
# Select from models in src/models/
model = "Epsilon_3" 
emb = "T5"             # ["T5", "ProstT5", "ProSE", "BERT", "ESM2-650M"]
emb_type = "global"    # ["global", "local"]
objective = ["interface_bin", 10, "avg", False, True, False]
# Specify location for storing ablation results. For no ablations set it to "".
ablations_dir = "" # "Ablations/"  # Uncomment while running ablations.
"""
Note:
interaction/interaction_bin refers to the contact map prediction task.
interface/interface_bin refers to the interface prediction task.
"_bin" indicates CG models.
objective is list of 4 elements:
    obj --> ["interaction", "interface", "interaction_bin", "interface_bin"]
    CG --> [1, 5, 10]
    pool --> "avg"/"max" -- only using avg.
    bin_post_proj --> bin embeddings post projection block -- deprecated.
    bin_input --> Set to true for CG tasks else False.
    single_output --> deprecated. Leave it to False.
"""

########################

data = {
    "Version": "5",
    "Embedding": f"{emb}",
    "Emb_type": emb_type,
    "Model": model,
    "System": str( system.strip() ),
    "Global_seed": 1,
    "conf": {
        "model_params": {
            "Model": model,
            "emb_size": 1024,
            # ["projection_dim" --> size of projected embedding,
            #  "projection_layer_type" --> refer to models/get_layers.py, 
            #  "bias" --> b in wx + b, 
            # "multiplier" --> scaling factor for auto3 proj_layer.
            #  "separate_proj_layer" --> use separate or common projection layer ["separate" or ""]]
            "projection_layer": [[128, "ln2", True, 1, ""]],
            "output_dim": 1,
            # ["aggregate", "conactenate", "interface"]
            #       aggregate --> ["add", "substract", "multiply", "op-od", "dot", "cosine"]
            #       concatenate --> ["vanilla", "conact"]
            #       interface --> ["avg1d", "avg2d", "lin", ""] - required for interface not for interaction task.
            "input_layer": ["op-od", "vanilla", "lin"],
            # For Monte carlo dropout. #samples to be taken. -- deprecated
            "num_samples": 0,
            # ["#US_layers", "#DS_layers", "num_blocks", "scale_factor"", 
            #               "hidden_block_type - vanilla/residual", 
            #               "residual_connection -- vanilla/addnorm/addactivnorm"]
            "num_hid_layers": [[0, 0, 0, 0, "vanilla", ""]],
            "bias": True,
            # [dropout1, dropout2, us_dropout, ds_dropout, mc_dropout (deprecated)]
            "dropouts": [[0.2, 0, 0, 0, 0]],
            # Supported - BNorm (BN), INorm (IN), LNorm (LN)
            "norm": [True, "LN"],
            # Must be a float value or None.
            "temperature": None, 
            # ["vanilla", "conf", "mc", "count_reg"]
            "output_layer": "vanilla",
            # [activation name, activation param]
            "activation1": [["elu", None]],
            # [activation name, apply activation or not]
            "activation2": ["sigmoid", True],
            "device": "cpu",
            "objective": objective
            },
        "dataset": {
            # For record purpose. Dataset has been pre-split into train:dev:test set.
            "train_set_size": 0.9,
            "dev_set_size": 0.05,
            "test_set_size": 0.05,
            "input_files": f"/data2/kartik/Disorder_Proteins/disobind_archive/Database/v_{dataset_version}/{emb}/{emb_type}-None/", 
            "train_file": f"Train_set_{emb_type}_v_{dataset_version}.npy",
            "dev_file": f"Dev_set_{emb_type}_v_{dataset_version}.npy",
            "test_file": f"Test_set_{emb_type}_v_{dataset_version}.npy",
            "output_path": f"../Models/{model}_Train/{ablations_dir}",
            "batch_size": 64,
            "batch_shuffle": [True, False, False]
            },
        "train_params": {
            "objective": objective, # Specify objective and bin_size.
            "emb": emb,
            "mask": [False, True],
            "num_metrics": [7, "global"],
            "Nruns": 1,
            # Look src/loss.py
            "loss": "se_loss",
            # Set weights to be used for loss calculation.
            "log_weight": [[0.9, 3]],
            # Supported -- Adam, AdamW, SGD.
            "optimizer":"AdamW",
            # If True, uses amsgrad variant of optimizer.
            "amsgrad": True,
            "weight_decay": [0.05],
            # calibration methods - beta-abm, platt, temp, None.
            "calibration": "beta-abm",
            # For gradient clipping.
            "max_norm": None,
            "learning_rate": [1e-4],
            "scheduler": {
            # If True, usea lr scheduler else not.
                "apply": True,
                # ["linear", "exp", "multistep", "cycliclr", "swa"]
                "name": "exp",
                "milestone": [None],
                "gamma": [0.93],
                "start_factor": None,
                "end_factor": None,
                "total_iters": None,
                "swa_start": None,
                "swa_lr": None,
                "base_lr": None,
                "step_size_up": None,
                "step_size_down": None,
            },
            # Length of the journey.
            "max_epochs": 25,
            # Threshold for classifying a logit into contact/non-contact.
            "contact_threshold": [0.5],
            "save_model": True,
            "model_path": f"model_{emb_type}",
            "optuna_trials": 0
            }
        }
    }

OmegaConf.save( config = data, f = "version_{}.yml".format( data["Version"] ) )

