# MLMI4 Group 10
## Setting up
To create a conda environment with `python==3.13` and install the packages required for this project, run the following code.
```python
conda create --prefix ./.conda python=3.13
conda activate ./.conda
pip install -r requirements.txt
```

## Testing with PyTest
To run the tests, activate the conda environment with `conda activate ./.conda` and run the following code.
```
python -m pytest -s
```

This should give output similar to (more extensive than) the following.

![ ](./figures/example_test_output.png)