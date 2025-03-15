# MLMI4 Group 10
## Setting up
To create a conda environment with `python==3.13` and install the packages required for this project, run the following code.
```python
conda create --prefix ./.conda python=3.13
conda activate ./.conda
pip install -r requirements.txt
```

## Extension: variance
To train:
```
run.py --extension=variance --pipeline=train --run_name=ext_variance
```
This creates a folder `./checkpoints/ext_variance` with the model checkpoints

To continue from a checkpoint, run
```
run.py --checkpoint ./checkpoints/ext_variance
```

To estimate the NLL from a checkpoint on the validation dataset, run
```
run.py --checkpoint ./checkpoints/ext_variance --pipeline=valid --reruns=5 --seed=42 
```
This will set the random seed with 42 and go through 5 passes of the validation dataset and calculate metrics on both model.pth and model_ema.pth, returning the mean and standard deviations


To generate and save some molecules, run
```
run.py --checkpoint=./checkpoints/ext_variance --pipeline=sample --batch-size=8 --num-samples=15
```

## Testing with PyTest
To run the tests, activate the conda environment with `conda activate ./.conda` and run the following code.
```
python -m pytest -s
```

This should give output similar to (more extensive than) the following.

![ ](./figures/example_test_output.png)