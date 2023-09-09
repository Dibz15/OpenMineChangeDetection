# Open Mine Change Detection


<p align="center">
  <img src="media/Iluka%20Western%20Australia_average_NDTCI.png" alt="Description of Image 1" width="48%" />
  <img src="media/Iluka%20Western%20Australia_hybrid_mask.png" alt="Description of Image 2" width="48%" />
</p>

## Overview

**Created by Austin Dibble as part of his MSc thesis research at University of Strathclyde.**

This repository is all the code that I have used in the experimentation and writing of my MSc dissertation titled "Characterising Open Cast Mining
from Satellite Data". 

For the three models which were used primarily for performance comparison on my proposed dataset, they each exist in their own submodule directory. The subdirectories were cloned from the original source repositories of the original projects.

- TinyCD:  [https://github.com/AndreaCodegoni/Tiny_model_4_CD](https://github.com/AndreaCodegoni/Tiny_model_4_CD)
- LSNet: [https://github.com/qaz670756/LSNet](https://github.com/qaz670756/LSNet)
- DDPM-CD: [https://github.com/wgcban/ddpm-cd](https://github.com/wgcban/ddpm-cd)

Minor modifications were performed on each model. Generally, the changes made were simply intended to create a PyTorch Lightning wrapper for each model so that training all of the architectures could be performed more modularly. 

## Data Archives:

In my dissertation, I have proposed the OMS2CD dataset which features hand-labelled images for change-detection in open-pit mining areas. Here is a direct download link to the dataset archive hosted on Google Drive: [OMS2CD download.](https://drive.google.com/file/d/1Kyle3U-lHQsj_zo7xO-GQJk_ZX9SmiKG/view?usp=drive_link)

I had also prepared an archive of data for case study examples. While they are not labelled, they may still prove useful: [Case study data archive link.](https://drive.google.com/file/d/1--NJ9XpXs4-PZarzDE3bMWiKX12Ot0DU/view?usp=sharing)

## Notebooks: 

All of the experimentation was performed inside a collection of Jupyter Notebooks, which are all within the `notebooks` subdirectory. Here is a list with a short description of each one that is relevant to my final dissertation, as written:

- `TinyCD_OMS2CD.ipynb`: Training notebook for TinyCD on OMS2CD.
- `LSNet_OMS2CD.ipynb`: Training notebook for LSNet on OMS2CD
- `DDPMCD_OMS2CD.ipynb`: Training notebook for DDPM-CD on OMS2CD.
- `SiteChangeModelling.ipynb`: A notebook which features the creation of many visualisations on OMS2CD and the case study data described in my dissertation.
- `GetSentinelData.ipynb`: A notebook which features the described process for downloading and processing the Sentinel-2 tiles using Google Earth Engine, both for OMS2CD and the case study sites.
- `CreateDatasetMapping.ipynb`: A short notebook which was used to generate the `mapping.csv` file in the OMS2CD dataset archive.
- `SubsetDataByManualMasks.ipynb`: A short notebook which was used to get a subset of the full OMS2CD data based on which files had been fully annotated with labels.
- `TransferDomainExample.ipynb`: A short notebook which demonstrates the performance of the models on OMS2CD when they had only been trained on other datasets (such as OSCD, OMCD, and LEVIR-CD).

Please note that the notebook were written to be used in Google Colab. As such, they download the OpenMineChangeDetection and add it as a submodule to the Python path. Therefore, all imports are performed with `OpenMineChangeDetection` as a separate module. If the notebooks are run as-is from within the repository on a local Jupyter Notebook instance, they will not operate correctly.

## Key Classes

The notebooks given above can be seen for usage examples, but the most important classes in this repository are given with a short description here. Our OMS2CD class is given as an example implementation for how to work with and load the proposed OMS2CD dataset.

- `datasets.OMS2CD`: Effectively a PyTorch Dataloader which can be used to load the OMS2CD dataset for training/inference.
- `datasets.OMS2CDDataModule`: A PyTorch Lightning datamodule wrapper around the OMS2CD dataloader class. It can be used directly with PyTorch Lightning.
- `ddpm_cd.ddpm_lightning.CD`: A PyTorch Lightning module wrapper around the DDPM-CD architecture classes. Can be used directly with the pl.Trainer.fit() function.
- `LSNet.LSNetLightning`: A PyTorch Lightning module wrapper around the LSNet architecture class. Can be used directly with the pl.Trainer.fit() function.
- `TinyCD.models.cd_lightning.ChangeDetectorLightningModule`: A PyTorch Lightning module wrapper around the TinyCD architecture class. Can be used directly with the pl.Trainer.fit() function.

## Site Data: 

The `site_data` subdirectory (first level) contains the GeoJSON data described in my dissertation. `active_sites_corrected.geojson` contains the locations used to create the OMS2CD dataset, and `site_data_start_2019_selected.geojson` contains the location data used for the case study comparisons from the Analysis and Appendix chapters.

The subdirectories within `site_data` contain a myriad of experimental results and visualisations which were used in comparing models and verifying the inference of the models on different locations.

## Notes on Code Written By Me:

- In `datasets`: `OMS2CD`, `OMCD`, `OSCD_Chipped` were written by me, with a few small portions copied from the `torchgeo` source repository (see the licences section below). Such copying was performed in good faith and follows the terms of the licence available under the source repository. In each file is given the appropriate licence notice. See Licences section below.
- Lightning class wrappers around the original models listed in the `Key Classes` section above. They are a mixture of my own modifications and the original training code given in the source repositories. Code reuse was performed in good faith following the terms of the licences in the original repositories. Each Lightning module file includes the appropriate licence notices. See the Licences section below.
- `ndtci.py`: Utility functions for calculating and visualising the proposed NDTCI metric.
- `visualisations.py`: A collection of visualisation functions that were used in my notebook scripts. All of the visualisation functions were also written by me, with reference to the corresponding library (matplotlib, seaborn, etc.) API documentation.

## Library Dependencies

The library imports seen in the code and notebooks assumes execution from a Google Colab environment, which includes all required dependencies besides those explicitly installed in the notebooks. If running the code outside of Google Colab, you may need to install the following: 
- PyTorch
- torchvision
- tqdm
- pandas
- NumPy
- matplotlib

The libraries used explicitly, which must be installed within Colab: 
- rasterio==1.3.8
- torchgeo==0.4.1
- tiler==0.5.7
- kornia==0.6.12
- lightning==1.9.5
- torchmetrics==0.11.4

The versions listed were fixed to ensure that the notebooks can be run in the future, without breaking API changes that can occur with newer versions. 

## Licences

Each of the model subdirectories which were adapted for this research have their own explicit licences. They are listed here. Please note that while this repository (and code explicitly written by me) is provided under an Open Software License 3.0, the individual licence of the listed subdirectories may grant different rights and have different limitations. Please consider them carefully.

- **TinyCD**: The licence provided reads "Code is released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors." 
  - Note: I have contacted the owner of the original TinyCD repository and they have given me explicit permission to use and extend the original source code for my research.
- **LSNet**: Licensed under Apache 2.0. 
  - Stated changes: Any changes made to the original LSNet repository are visible in the commit history of this history. In general, the changes made were as minimal as possible. Some modifications were intended to improve module importing when using the code in a Jupyter Notebook. Other changes include adding a PyTorch Lightning wrapper class of the original LSNet training code, and making configuration file changes.
- **DDPM-CD**: Licensed under an MIT License.

Additionally, under `datasets` I have adapted some code from the [torchgeo](https://github.com/microsoft/torchgeo) and [tiler](https://github.com/the-lay/tiler) libraries for loading the OMS2CD, OMCD, and OSCD datasets. Both libraries are available under an MIT License.

Copyrights:
- Sentinel Images:
  - Data provided by the European Space Agency.
  - This OMS2CD dataset contains modified Copernicus data from 2018-2020. Original Copernicus Sentinel Data available from the European Space Agency (https://sentinel.esa.int).