# XAI-Explainability_of_FND_Models
--------------------------------------------------------------------------------------------------------------
Author : MANAI MORTADHA

# ABOUT :

This repository aims to explain state-of-the-art Fake News Detection models.
Huggingface module is intended for explaining only-text-based fake news detection models in Transformers.
GNNFakeNews module is an attempt to explain GNNs that are hybrid models for fake news
detection using GNNExplainer and latent space exploration.


In order to run a notebook, you would need to create two virtual environments one for Huggingface and one for
GNNFakeNews in respective directories and should activate the related virtual environment before running the notebooks.

- Models in Huggingface: DistilRoBERTa
- Models in GNNFakeNews: GCN, GAT, GraphSAGE, GCNFN

Setup for Huggingface:
----------------------
In the root folder of this project, type

- `cd Huggingface`
- `conda env create -f environment.yml`
- `conda activate huggingface2`
  After these commands run successfully, any notebook (except the notebooks under deprecated folder) should without any
  issues.

Setup for GNNFakeNews:
----------------------
In the root folder of this project, type

- `cd GNNFakeNews`
- `conda env create -f environment.yml`
- `conda activate gnnfakenews`

You might need to install PyG manually, [here](https://github.com/pyg-team/pytorch_geometric) are the instructions.
After these commands run successfully, any notebook (except the notebooks under deprecated folder) should run without
any issues.#

! Always make sure that the active virtual environment is the one that you should work in.

