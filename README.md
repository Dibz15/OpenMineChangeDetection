# Open Mine Change Detection

## Overview

This repository is all the code that I have used in the experimentation and writing of my MSc dissertation titled "Characterising Open Cast Mining
from Satellite Data". 

For the three models which were used primarily for performance comparison on my proposed dataset, they each exist in their own submodule directory. The subdirectories were cloned from the original source repositories of the original projects.

- TinyCD:  [https://github.com/AndreaCodegoni/Tiny_model_4_CD](https://github.com/AndreaCodegoni/Tiny_model_4_CD)
- LSNet: [https://github.com/qaz670756/LSNet](https://github.com/qaz670756/LSNet)
- DDPM-CD: [https://github.com/wgcban/ddpm-cd](https://github.com/wgcban/ddpm-cd)

Minor modifications were performed on each model. Generally, the changes made were simply to create a PyTorch Lightning wrapper for each model so that training all of the architectures could be performed more modularly. 

## Notebooks: 

All of the experimentation was performed inside a collection of Jupyter Notebooks, which are all within the `notebooks` subdirectory. Here is a list with a short description of each one that is relevant to my final dissertation, as written:

- `TinyCD_OMS2CD.ipynb`:
- `LSNet_OMS2CD.ipynb`:
- `DDPMCD_OMS2CD.ipynb`:
- `SiteChangeModelling.ipynb`:
- `GetSentinelData.ipynb`:

## Key Classes

The notebooks given above can be seen for usage examples, but the most important classes in this repository are given with a short description here. Our OMS2CD class is given as an example implementation for how to work with and load the proposed OMS2CD dataset.

- `datasets.OMS2CD`: 
- `datasets.OMS2CDDataModule`:
- `ddpm_cd.ddpm_lightning.CD`:
- `LSNet.LSNetLightning`:
- `TinyCD.models.cd_lightning.ChangeDetectorLightningModule`:

## Site Data: 

The `site_data` subdirectory (first level) contains the GeoJSON data described in my dissertation. `active_sites_corrected.geojson` contains the locations used to create the OMS2CD dataset, and `site_data_start_2019_selected.geojson` contains the location data used for the case study comparisons from the Analysis and Appendix chapters.

The subdirectories within `site_data` contain a myriad of experimental results and visualisations which were used in comparing models and verifying the inference of the models on different locations.

## Licenses

Each of the model subdirectories which were adapted for this research have their own explicit licenses. They are listed here. Please note that while this repository is provided under an Apache 2.0 License, the individual license of the listed subdirectories may grant different rights and have different limitations. Please consider them carefully.

- **TinyCD**: The license provided reads "Code is released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors." 
  - Note: I have contacted the owner of the original TinyCD repository and they have given me explicit permission to use and extend the original source code for my research.
- **LSNet**: Licensed under Apache 2.0. 
  - Stated changes: Any changes made to the original LSNet repository are visible in the commit history of this history. In general, the changes made were as minimal as possible. Some modifications were intended to improve module importing when using the code in a Jupyter Notebook. Other changes include adding a PyTorch Lightning wrapper class of the original LSNet training code, and making configuration file changes.
- **DDPM-CD**: Licensed under an MIT License.