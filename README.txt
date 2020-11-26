This folder contains code for the paper 'BoMuDA: Boundless Multi-Source Unsupervised Semantic Segmentation
in Unconstrained Environments'

Code structure:

'dataset' folder - dataloaders, image lists
'model' folder - network architectures
'utils' folder - additional functions
eval_idd_openset.py - evaluation script for IDD, for the overall algorithm
eval_idd_ensemble.py - evaluation script for model outputs ensembled from step 1 and step 2
train_singlesourceDA.py - training script for single source DA
train_bddbase_multi3source_furtheriterations.py - training script for step 1 
train_multi3source_combinedbddbase.py - training script for step 2
train_openset.py - training script for boundless DA module