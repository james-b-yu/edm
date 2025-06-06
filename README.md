# MLMI4 Group 10
# Setting up
To create a conda environment with `python==3.13` and install the packages required for this project, run the following code.
```python
conda create --prefix ./.conda python=3.13
conda activate ./.conda
pip install -r requirements.txt
```

# Calculating metrics on pre-trained weights
To calculate an estimate of NLL on the pretrained weights using the test set, run the following code:
```
python run.py --pipeline=test --seed=42 --reruns=3 --checkpoint=./pretrained/<model-name>
```
where `<model-name>` is any one of `'vanilla_with_h'`, `'vanilla_without_h'`, `'variance_with_h'` or `'variance_without_h'`

The extension and model hyperparameters are automatically activated based on the contents of `args.pkl` in the checkpoint folder.

You may specify `--pipeline=valid` if you would like to calculate metrics using the validation set.

TODO: also calculate stability metrics, etc.

# Sampling from the model
To sample from pretrained models, use the following command.
```
python run.py --pipeline=sample --seed=42 --num-samples=<num-samples> --checkpoint=./pretrained/<model-name>
```
where `<num-samples>` is the number of samples to create, e.g. 1000 and `<model-name>` is any one of `'vanilla_with_h'`, `'vanilla_without_h'`, `'variance_with_h'` or `'variance_without_h'`.

# Training the model
To train a model from scratch, run:
```
python run.py --pipeline=train --seed=42 --extension=<extension> --dataset=<dataset> --run-name=<run-name>
```
where `extension` can be one of `vanilla` or `variance`, and `dataset` can be one of `qm9` or `qm9_no_h`.

This runs for 1300 epochs by default and saves checkpoints for every epoch in `./checkpoints/<run-name>`.

To load from a checkpoint and continue training, run
```
python run.py --pipeline=train --checkpoint="./checkpoints/<run-name>" --start-epoch=<start-epoch>
```
where `start-epoch` is zero-indexed and equals the epoch after the last full epoch currently trained at -- this is purely used for bookkeeping but the code itself does not automatically detect which epoch number to start at so you must specify this manually.

You do not need to specify the extension or any other hyperparamters, as these are automatically activated from `args.pkl` located in the checkpoint folder.