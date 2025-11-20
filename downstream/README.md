# Downstream

## Clone repository
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

## Prepare pre-trained weights
Please download the [pre-trained weights](https://drive.google.com/file/d/1kEhA5ViCCwfnYtPbOn4rkWlikK1C4emb/view?usp=sharing) of SOTA models, and unzip it under /downstream/.

## Quick demo
We have also included a quick demo to support training, testing, and visualization.
Please follow the steps:
1. Download the preprocessed [demo data](https://drive.google.com/file/d/17mio1t455qQIyVDgtBBfhEK4kKAtwCDG/view?usp=sharing) (Thanks for [INbreast dataset](https://www.sciencedirect.com/science/article/abs/pii/S107663321100451X)), and put it under /downstream/Quick_demo/.
2. Download the [pre-trained weight](https://drive.google.com/file/d/1ryAhGluZls7Oq4ELHVj4hcq1TM8gHGO5/view?usp=sharing) for VQA test, and put it under /downstream/Quick_demo/VQA/.
3. ```shell
   cd Quick_demo
   ```

### Classification
1. Train the model:
   ```shell
   python Classification/main.py
   ```
2. Evaluate the model:
   ```shell
   python Classification/eval.py
   ```

### Segmentation
1. Train the model:
   ```shell
   python Segment/main.py
   ```
2. Evaluate the model:
   ```shell
   python Segment/eval.py
   ```

### Detection
1. Train the model:
   ```shell
   python Detection/main.py
   ```
2. Evaluate the model:
   ```shell
   python Detection/eval.py
   ```

### VQA
1. Evaluate the model:
   ```shell
   python VQA/eval.py
   ```

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

