This folder contains code for the paper 'BoMuDA: Boundless Multi-Source Unsupervised Semantic Segmentation
in Unconstrained Environments'

Code structure:

'dataset' folder - dataloaders, image lists
'model' folder - network architectures
'utils' folder - additional functions
eval_idd_openset.py - evaluation script for IDD, for the overall algorithm
train_singlesourceDA.py - training script for single source DA
train_bddbase_multi3source_furtheriterations.py - training script for Alt-Inc
train_openset.py - training script for boundless DA module