# Datasets prepare

## Download datasets
### Datasets-Links:
Datasets downloading URL:
    
| Dataset Name | Link | Access |
|-----|---------------|--------|
| BMCD | https://zenodo.org/records/5036062 | Open Access |
| CBIS-DDSM | https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset | Open Access |
| CDD-CESM | https://www.kaggle.com/datasets/krinalkasodiya/new-cdd-cesm-classification-data | Open Access |
| DMID | https://figshare.com/articles/dataset/_b_Digital_mammography_Dataset_for_Breast_Cancer_Diagnosis_Research_DMID_b_DMID_rar/24522883 | Open Access |
| INbreast | https://www.kaggle.com/datasets/tommyngx/inbreast2012 | Open Access |
| MIAS | https://www.kaggle.com/datasets/kmader/mias-mammography | Open Access |
| CSAW-M | https://figshare.scilifelab.se/articles/dataset/CSAW-M_An_Ordinal_Classification_Dataset_for_Benchmarking_Mammographic_Masking_of_Cancer/14687271 | Credentialed Access |
| KAU-BCMD | https://www.kaggle.com/datasets/asmaasaad/king-abdulaziz-university-mammogram-dataset?select=Birad5 | Open Access |
| VinDr-Mammo | https://www.physionet.org/content/vindr-mammo/1.0.0/ | Credentialed Access |
| RSNA | https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data | Open Access |
| EMBED | https://registry.opendata.aws/emory-breast-imaging-dataset-embed/ | Credentialed Access |
| DBT-Test | https://www.cancerimagingarchive.net/collection/breast-cancer-screening-dbt/ | Open Access |
| LAMIS | https://github.com/LAMISDMDB/LAMISDMDB_Sample | Credentialed Access |
| MM | https://data.mendeley.com/datasets/fvjhtskg93/1 | Open Access |
| NLBS | https://www.frdr-dfdr.ca/repo/dataset/cb5ddb98-ccdf-455c-886c-c9750a8c34c2 | Open Access |

## Prepare classification datasets
After downloaded datasets above, you have to use the correspoding processing code for it. Remember to change the dataset link in the code!!!

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

## Prepare segmentation\detection datasets
After downloaded datasets above, you have to use the correspoding processing code for it. Remember to change the dataset link in the code!!!

### Processing Dataset Codes and Files Linking:

| Dataset Name | Process Dataset Code |
|--------------|----------------------|
| CBIS-DDSM | https://github.com/fuxianghuang1/VersaMammo/blob/main/datapre/preprocess/CBIS-DDSM.ipynb |
| INbreast | https://github.com/fuxianghuang1/VersaMammo/blob/main/datapre/preprocess/INbreast.ipynb |
| VinDr-Mammo | https://github.com/fuxianghuang1/VersaMammo/blob/main/datapre/preprocess/VinDr-Mammo.ipynb |

## Prepare VQA datasets
Please prepare the dataset according to [MammoVQA](https://github.com/PiggyJerry/MammoVQA), and put the dataset's json files under /downstream/VQA/.

# Downstream

## Clone repository
```shell
git clone https://github.com/fuxianghuang1/VersaMammo.git

conda create -n downstream python==3.9
conda activate downstream

python -m pip install -r requirements.txt
```

## Prepare pre-trained weights
Please download the [pre-trained weights](https://drive.google.com/file/d/1kEhA5ViCCwfnYtPbOn4rkWlikK1C4emb/view?usp=sharing) of SOTA models, and unzip it under /downstream/.

## Training
For all the downstream tasks, you can use the following command to train the models (please modify the downstream_task to the corresponding task):
```shell
bash /downstream/downstream_task/bash.sh
```

## Testing
For all the downstream tasks, you can use the following command to test the models (please modify the downstream_task to the corresponding task):
```shell
bash /downstream/downstream_task/eval.sh
```

