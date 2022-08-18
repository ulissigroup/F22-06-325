import torch
import matplotlib.pyplot as plt
torch.cuda.is_available()

from ocpmodels.datasets import TrajectoryLmdbDataset, SinglePointLmdbDataset

# TrajectoryLmdbDataset is our custom Dataset method to read the lmdbs as Data objects. Note that we need to give the path to the folder containing lmdbs for S2EF
dataset = TrajectoryLmdbDataset({"src": "data/s2ef/train_100/"})

print("Size of the dataset created:", len(dataset))
print(dataset[0])

# + colab={"base_uri": "https://localhost:8080/"} id="pD5B_TymoJ8S" outputId="72b21c2a-9472-4b08-afe9-c1bd28a5b399"
data = dataset[0]
data

# + colab={"base_uri": "https://localhost:8080/"} id="rL4u0glIoL8h" outputId="a29c8dfc-617f-48fa-9195-e851b23033e1"
energies = torch.tensor([data.y for data in dataset])
energies

# + colab={"base_uri": "https://localhost:8080/", "height": 737} id="mkOm2roAoNY2" outputId="aed9b4de-99de-49ab-a21c-3a372166747a"
plt.hist(energies, bins = 50)
plt.yscale("log")
plt.xlabel("Energies")
plt.show()

from ocpmodels.trainers import EnergyTrainer
from ocpmodels.datasets import SinglePointLmdbDataset
from ocpmodels import models
from ocpmodels.common import logger
from ocpmodels.common.utils import setup_logging
setup_logging()

import numpy as np
import copy
import os

train_src = "data/is2re/train_100/data.lmdb"
val_src = "data/is2re/val_20/data.lmdb"

train_dataset = SinglePointLmdbDataset({"src": train_src})

energies = []
for data in train_dataset:
  energies.append(data.y_relaxed)

mean = np.mean(energies)
stdev = np.std(energies)

task = {
  "dataset": "single_point_lmdb",
  "description": "Relaxed state energy prediction from initial structure.",
  "type": "regression",
  "metric": "mae",
  "labels": ["relaxed energy"],
}

# Model
model = {
    'name': 'gemnet_t',
    "num_spherical": 7,
    "num_radial": 64,
    "num_blocks": 5,
    "emb_size_atom": 256,
    "emb_size_edge": 512,
    "emb_size_trip": 64,
    "emb_size_rbf": 16,
    "emb_size_cbf": 16,
    "emb_size_bil_trip": 64,
    "num_before_skip": 1,
    "num_after_skip": 2,
    "num_concat": 1,
    "num_atom": 3,
    "cutoff": 6.0,
    "max_neighbors": 50,
    "rbf": {"name": "gaussian"},
    "envelope": {
      "name": "polynomial",
      "exponent": 5,
    },
    "cbf": {"name": "spherical_harmonics"},
    "extensive": True,
    "otf_graph": False,
    "output_init": "HeOrthogonal",
    "activation": "silu",
    "scale_file": "configs/s2ef/all/gemnet/scaling_factors/gemnet-dT.json",
    "regress_forces": False,
    "direct_forces": False,
}
# Optimizer
optimizer = {
    'batch_size': 1,         # originally 32
    'eval_batch_size': 1,    # originally 32
    'num_workers': 2,
    'lr_initial': 1.e-4,
    'optimizer': 'AdamW',
    'optimizer_params': {"amsgrad": True},
    'scheduler': "ReduceLROnPlateau",
    'mode': "min",
    'factor': 0.8,
    'patience': 3,
    'max_epochs': 1,         # used for demonstration purposes
    'ema_decay': 0.999,
    'clip_grad_norm': 10,
    'loss_energy': 'mae',
}
# Dataset
dataset = [
  {'src': train_src,
   'normalize_labels': True,
   'target_mean': mean,
   'target_std': stdev,
  }, # train set 
  {'src': val_src}, # val set (optional)
]

energy_trainer = EnergyTrainer(
    task=task,
    model=copy.deepcopy(model), # copied for later use, not necessary in practice.
    dataset=dataset,
    optimizer=optimizer,
    identifier="IS2RE-example",
    run_dir="./", # directory to save results if is_debug=False. Prediction files are saved here so be careful not to override!
    is_debug=False, # if True, do not save checkpoint, logs, or results
    print_every=5,
    seed=0, # random seed to use
    logger="tensorboard", # logger of choice (tensorboard and wandb supported)
    local_rank=0,
    amp=True, # use PyTorch Automatic Mixed Precision (faster training and less memory usage)    
)

energy_trainer.model
energy_trainer.train()

