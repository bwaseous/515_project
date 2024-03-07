# 515_project
AMATH 515 Winter 2024 project on testing stochastic gradient descent algorithms and learning rate schedulers by Avinash Joshi. 

This project expects a CUDA 12.0 compatible NVIDIA GPU for full utilization.

## Installing the project

In a Python 3.10 environment, either run

```
pip install -r gpu_reqs.txt
```

or via `conda` with

```
conda env create -f env.yaml
```

If the environment is working, the following command should output True:

```python
import torch

print(torch.cuda.is_available())
```

## Data

The data results can be found in the `data` folder and can be loaded for inspection using the plotting section found below the training section of each dataset.