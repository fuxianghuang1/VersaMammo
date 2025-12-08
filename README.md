
## Contents

- [Installation](#Installation)
  - [Pretraining](#pretraining-prerequisites)
  - [Downstream task](#downstream-task-prerequisites)

- [Data Preparation](#data-preparation)
  - [Download dataset link](#download-dataset-link)
  - [Prepare segmentation and detection datasets](#prepare-segmentation-and-detection-datasets)  
  - [Prepare retrieval datasets](#prepare-retrieval-datasets)
  - [Prepare classification datasets](#prepare-classification-datasets)
  - [Prepare VQA datasets](#prepare-vqa-datasets)

- [Run](#run)
  - [Pretraining](#pretraining)
  - [Downstream task](#downstream-task)

- [Acknowledgements](#acknowledgements)

# Installation
## Pretraining prerequisites
```shell
git clone https://github.com/fuxianghuang1/VersaMammo.git
cd VersaMammo/pretraining

# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate versamammo_pretrain

# If you have base CUDA environment
pip install -r requirements.txt

```

## Downstream task prerequisites
```shell
git clone https://github.com/fuxianghuang1/VersaMammo.git
cd VersaMammo/downstream

conda create -n downstream python==3.9
conda activate downstream
# for pip
python -m pip install -r requirements.txt

# for conda
conda env update -f environment.yml
```

# Data Preparation

## Download dataset link
Datasets downloading URL:
    
| Dataset Name | Dataset Link | Paper Link | Access |
|--------------|--------------|------------|--------|
| BMCD | [Link](https://zenodo.org/records/5036062) | [Digital subtraction of temporally sequential mammograms for improved detection and classification of microcalcifications](https://link.springer.com/content/pdf/10.1186/s41747-021-00238-w.pdf) | Open Access |
| CBIS-DDSM | [Link](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) | [A curated mammography data set for use in computer-aided detection and diagnosis research](https://www.nature.com/articles/sdata2017177.pdf) | Open Access |
| CDD-CESM | [Link](https://www.kaggle.com/datasets/krinalkasodiya/new-cdd-cesm-classification-data) | [Categorized contrast enhanced mammography dataset for diagnostic and artificial intelligence research](https://www.nature.com/articles/s41597-022-01238-0.pdf) | Open Access |
| DMID | [Link](https://figshare.com/articles/dataset/_b_Digital_mammography_Dataset_for_Breast_Cancer_Diagnosis_Research_DMID_b_DMID_rar/24522883) | [Digital mammography dataset for breast cancer diagnosis research (dmid) with breast mass segmentation analysis](https://link.springer.com/article/10.1007/s13534-023-00339-y) | Open Access |
| INbreast | [Link](https://www.kaggle.com/datasets/tommyngx/inbreast2012) | [Inbreast: toward a full-field digital mammographic database](https://repositorio.inesctec.pt/server/api/core/bitstreams/6bc3ba6a-1220-413d-9ffe-a89acb92652b/content) | Open Access |
| MIAS | [Link](https://www.kaggle.com/datasets/kmader/mias-mammography) | The mammographic images analysis society digital mammogram database | Open Access |
| CSAW-M | [Link](https://figshare.scilifelab.se/articles/dataset/CSAW-M_An_Ordinal_Classification_Dataset_for_Benchmarking_Mammographic_Masking_of_Cancer/14687271) | [Csaw-m: An ordinal classification dataset for benchmarking mammographic masking of cancer](https://arxiv.org/pdf/2112.01330) | Credentialed Access |
| KAU-BCMD | [Link](https://www.kaggle.com/datasets/asmaasaad/king-abdulaziz-university-mammogram-dataset?select=Birad5) | [King abdulaziz university breast cancer mammogram dataset (kau-bcmd)](https://www.mdpi.com/2306-5729/6/11/111) | Open Access |
| VinDr-Mammo | [Link](https://www.physionet.org/content/vindr-mammo/1.0.0/) | [Vindr-mammo: A large-scale benchmark dataset for computer-aided diagnosis in full-field digital mammography](https://www.nature.com/articles/s41597-023-02100-7.pdf) | Credentialed Access |
| RSNA | [Link](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data) | [RSNA: Radiological Society of North America. Rsna screening mammography breast cancer detection ai challenge](https://www.rsna.org/rsnai/ai-image-challenge/screening-mammography-breast-cancer-detection-ai-challenge) | Open Access |
| EMBED | [Link](https://registry.opendata.aws/emory-breast-imaging-dataset-embed/) | [The emory breast imaging dataset (embed): A racially diverse, granular dataset of 3.4 million screening and diagnostic mammographic images](https://pubs.rsna.org/doi/pdf/10.1148/ryai.220047) | Credentialed Access |
| DBT-Test | [Link](https://www.cancerimagingarchive.net/collection/breast-cancer-screening-dbt/) | [ Detection of masses and architectural distortions in digital breast tomosynthesis: a publicly available dataset of 5,060 patients and a deep learning model](https://arxiv.org/pdf/2011.07995) | Open Access |
| LAMIS | [Link](https://github.com/LAMISDMDB/LAMISDMDB_Sample) | [Lamis-dmdb: A new full field digital mammography database for breast cancer ai-cad researches](https://www.sciencedirect.com/science/article/abs/pii/S1746809423012569) | Credentialed Access |
| MM | [Link](https://data.mendeley.com/datasets/fvjhtskg93/1) | [Mammogram mastery: a robust dataset for breast cancer detection and medical education](https://www.sciencedirect.com/science/article/pii/S2352340924006000) | Open Access |
| NLBS | [Link](https://www.frdr-dfdr.ca/repo/dataset/cb5ddb98-ccdf-455c-886c-c9750a8c34c2) | [Full field digital mammography dataset from a population screening program](https://www.nature.com/articles/s41597-025-05866-0.pdf) | Open Access |

After downloaded datasets above, you have to use the correspoding processing code for it. Remember to change the dataset link in the code!!!

## Prepare segmentation and detection datasets
### Processing Dataset Codes and Files Linking:

| Dataset Name | Process Dataset Code |
|--------------|----------------------|
| CBIS-DDSM | https://github.com/fuxianghuang1/VersaMammo/blob/main/datapre/preprocess/CBIS-DDSM.ipynb |
| INbreast | https://github.com/fuxianghuang1/VersaMammo/blob/main/datapre/preprocess/INbreast.ipynb |
| VinDr-Mammo | https://github.com/fuxianghuang1/VersaMammo/blob/main/datapre/preprocess/VinDr-Mammo.ipynb |

## Prepare retrieval datasets

## Prepare classification datasets
### Processing Dataset Codes and Files Linking:

| Dataset Name | Process Dataset Code |
|--------------|----------------------|
| BMCD | https://github.com/fuxianghuang1/VersaMammo/blob/main/datapre/preprocess/BMCD.ipynb |
| CBIS-DDSM | https://github.com/fuxianghuang1/VersaMammo/blob/main/datapre/preprocess/CBIS-DDSM.ipynb |
| CDD-CESM | https://github.com/fuxianghuang1/VersaMammo/blob/main/datapre/preprocess/CDD-CESM.ipynb |
| CMMD | https://github.com/fuxianghuang1/VersaMammo/blob/main/datapre/preprocess/CMMD.ipynb |
| CSAW-M | https://github.com/fuxianghuang1/VersaMammo/blob/main/datapre/preprocess/CSAW-M.ipynb |
| DBT | https://github.com/fuxianghuang1/VersaMammo/blob/main/datapre/preprocess/DBT.ipynb |
| DMID | https://github.com/fuxianghuang1/VersaMammo/blob/main/datapre/preprocess/DMID.ipynb |
| INbreast | https://github.com/fuxianghuang1/VersaMammo/blob/main/datapre/preprocess/INbreast.ipynb |
| KAU-BCMD | https://github.com/fuxianghuang1/VersaMammo/blob/main/datapre/preprocess/KAU-BCMD.ipynb |
| LAMIS | https://github.com/fuxianghuang1/VersaMammo/blob/main/datapre/preprocess/LAMIS.ipynb |
| MIAS | https://github.com/fuxianghuang1/VersaMammo/blob/main/datapre/preprocess/MIAS.ipynb |
| MM | https://github.com/fuxianghuang1/VersaMammo/blob/main/datapre/preprocess/MM.ipynb |
| NLBS | https://github.com/fuxianghuang1/VersaMammo/blob/main/datapre/preprocess/NLBS.ipynb |
| VinDr-Mammo | https://github.com/fuxianghuang1/VersaMammo/blob/main/datapre/preprocess/VinDr-Mammo.ipynb |


## Prepare VQA datasets
Please prepare the dataset according to [MammoVQA](https://github.com/PiggyJerry/MammoVQA), and put the dataset's images under /datapre/VQA_data, dataset's json files under /downstream/VQA/.

## You will have the following structure:
````
VersaMammo
|--datapre
   |--classification_data
      |--INbreast
         |--Train
            |--20586934
               |--img.jpg
               |--info_dict.npy
            ...
         |--Eval
         |--Test
      |--BMCD
      ...
   |--segdetdata
      |--INbreast
         |--Train
            |--20586934
               |--bboxes.npy
               |--img.jpg
               |--mask.png
            ...
         |--Eval
         |--Test
      |--CBIS-DDSM
      ...
   |--VQA_data
      |--INbreast
         |--20586934
            |--img.jpg
            |--info_dict.npy
         ...
      |--BMCD
      ...
   ...
````

# Run
## Pretraining 

## Downstream task
If you want to train or test all models (including MedSAM, LVM-Med, Mammo-CLIP, MAMA, and our VersaMammo), please download the corresponding [pre-trained weights](https://drive.google.com/file/d/14E1eQxjrbU-U_7sHksiGZjUtoge4-I33/view?usp=sharing), and unzip it under /downstream/.

### Training
For all the downstream tasks, you can use the following command to train the models (please replace `downstream_task` with the name of the corresponding downstream task, e.g., `Segment`, `Detection`, `classification`, `VQA`, `Multi_Classification`):
```shell
bash /downstream/[downstream_task]/bash.sh
```

### Testing
For all the downstream tasks, you can use the following command to test the models (please replace `downstream_task` with the name of the corresponding downstream task, e.g., `Segment`, `Detection`, `classification`, `VQA`, `Multi_Classification`):
```shell
bash /downstream/[downstream_task]/eval.sh
```

### Quick demo
If you want to quickly test the VersaMammo, we have also included a quick demo to support training and testing.
Please follow the steps:
1. If you have already downloaded the pre-trained weights of all models and have the /downstream/Sotas directory, you can skip Step 1. If not, please download the [pre-trained weights](https://drive.google.com/file/d/1HmEzoJDs99-t6_mUnrjnkcY8nTJ8WeVp/view?usp=sharing) of VersaMammo and place them under /downstream/Sotas/.
2. Download the preprocessed [demo data](https://drive.google.com/file/d/17mio1t455qQIyVDgtBBfhEK4kKAtwCDG/view?usp=sharing) (Thanks for [INbreast dataset](https://www.sciencedirect.com/science/article/abs/pii/S107663321100451X)), and put it under /downstream/Quick_demo/.
3. Download the [pre-trained weight](https://drive.google.com/file/d/1ryAhGluZls7Oq4ELHVj4hcq1TM8gHGO5/view?usp=sharing) for VQA test, and put it under /downstream/Quick_demo/VQA/.
4. ```shell
   cd Quick_demo
   ```

#### Classification
1. Train the model:
   ```shell
   python Classification/main.py
   ```
2. Evaluate the model:
   ```shell
   python Classification/eval.py
   ```

#### Segmentation
1. Train the model:
   ```shell
   python Segment/main.py
   ```
2. Evaluate the model:
   ```shell
   python Segment/eval.py
   ```

#### Detection
1. Train the model:
   ```shell
   python Detection/main.py
   ```
2. Evaluate the model:
   ```shell
   python Detection/eval.py
   ```

#### VQA
1. Evaluate the model:
   ```shell
   python VQA/eval.py
   ```




