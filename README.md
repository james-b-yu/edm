# MLMI4 Group 10
# Setting up
To create a conda environment with `python==3.13` and install the packages required for this project, run the following code.
```python
conda create --prefix ./.conda python=3.13
conda activate ./.conda
pip install -r requirements.txt
```

# Metrics on pre-trained weights
To calculate an estimate of NLL on the pretrained weights using the test set, run the following code:
```
python run.py --pipeline=test --seed=42 --reruns=5 --checkpoint=./pretrained/<extension_name>
```
`<extension-name>` is any one of `'vanilla-with-h'`, `'vanilla-without-h'`, `'variance-with-h'` or `'variance-without-h'`,

The following table summarises the result of running this command:
|**Estimate of NLL**|No extensions (vanilla)|Learning variance|
|---|----|---|
|With hydrogens|`-111.44 (0.98)`|`-121.64 (0.69)`|
|Without hydrogens|`-22.96 (0.71)`|TODO|

The extension and model hyperparameters are automatically activated based on the contents of `args.pkl` in the checkpoint folder.

You may specify `--pipeline=valid` if you would like to calculate metrics using the validation set.

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