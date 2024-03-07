# HER2 Discordance and Rare FISH Patterns
This study evaluates efficacy of HER2-targeted therapy in cases with rare in situ hybridization patterns and discordant immunohistochemistry

<img src="https://github.com/fmhoward/HER2Discordance/blob/main/her2_subgroups_pcr.png?raw=true" width="800">

## Attribution
If you use this code in your work or find it helpful, please consider citing our repository (paper is pending).
```
@article{zhang_her2discordance_2024,
	title = {Efficacy of HER2-targeted therapy in cases with rare in situ hybridization patterns and discordant immunohistochemistry},
	author = {Zhang, Qianchen and Freeman, Jincong and Zhao, Fangyuan and Chen, Nan and Nanda, Rita and Huo, Dezheng and Howard, Frederick M.},
}
```

## Installation
The associated Jupyter notebook can be downloaded and all relevant code is provided within the notebook.

Requirements:
* python 3.9.12
* pandas 1.0.5
* numpy 1.19.0
* tableone 0.7.10
* lifelines 0.26.3

## Overview
The notebook is divided into sections to replicate all components of our paper.

The first section entitled 'Loading Data from NCDB' loads appropriate cases from the NCDB. This code requires the NCDB PUF (the 2020 version was used in this study), and the local file path of the NCDB PUF must be updated in this section of the code.  

The second section, entitled 'Demographic Analysis', creates demographic tables for all cohorts analyzed in this study.

The third section, entitled 'Neoadjuvant Cohort Analysis', was used for comparison of pCR rates between HER2+ subgroups and HER2- controls as well as immunotherapy-untreated cases from these subgroups.

The fourth section, 'Survival Cohort Analysis', was used for comparison of overall survival between HER2+ subgroups and HER2- controls as well as immunotherapy-untreated cases from these subgroups.
