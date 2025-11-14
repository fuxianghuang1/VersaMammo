# Downstream

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
