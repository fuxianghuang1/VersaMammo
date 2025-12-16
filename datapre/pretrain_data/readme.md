## 📂 Pretraining Data

Place your pretraining datasets in this directory. These datasets may also be used for downstream task fine-tuning.

### **Configuration**
- **`predata.csv`**: This file contains the metadata for all pretraining data (Stage 1 & Stage 2).
  - **Important**: Before running the code, replace the placeholder path `../dataset` in this file with your **actual pretraining data path**.

### **Stage 2 Preparation**
If you are conducting Stage 2 training, first download the image features extracted from Stage 1 and place them here:
- [Download Features (Selected Data)](https://drive.google.com/file/d/1Diu1aS5Y5xIol8llEdnSe-6415hiQuaD/view?usp=drive_link)
- [Download Features (Full Pretraining Data)](https://drive.google.com/file/d/1lfGztm0wi0NoMloD1FIZJpBzNjBMXxpg/view?usp=drive_link)