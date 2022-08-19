---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"id": "dzeHYa5GCxN7"}

`````{note}
This project was adapted from a tutorial on Open Catalyst Project Models as part of a ClimateChangeAI tutorial available here:
https://colab.research.google.com/drive/1oGZcrakB4Pbj8Xq74lSvcRDUHw9L-Dh5?usp=sharing#scrollTo=FIiwpALzBKaH

Prof. Ulissi was one of the authors of that tutorial, and the tutorial was released openly under an MIT license. This modified content is released under the CC-by-4 license in the main repository.

Full list of contributing authors to the original tutorial:
* [Muhammed Shuaibi](https://mshuaibii.github.io/), CMU, mshuaibi@andrew.cmu.edu
* [Abhishek Das](https://abhishekdas.com/), FAIR, abhshkdz@fb.com 
* [Adeesh Kolluru](https://adeeshkolluru.github.io/), CMU, akolluru@andrew.cmu.edu
* [Brandon Wood](https://wood-b.github.io/), NERSC, bwood@lbl.gov 
* [Janice Lan](https://www.linkedin.com/in/janice-lan), FAIR, janlan@fb.com
* [Anuroop Sriram](https://www.linkedin.com/in/anuroopsriram), FAIR, anuroops@fb.com
* [Zachary Ulissi](https://ulissigroup.cheme.cmu.edu/), CMU, zulissi@andrew.cmu.edu
* [Larry Zitnick](http://larryzitnick.org/), FAIR, zitnick@fb.com
`````

+++

![OC20 ask](./oc20_tasks.png)

+++ {"tags": []}

# Using ML to Predict Catalyst Properties

Prof. Ulissi works on building computational datasets and machine learning models to predict the activity and selectivity of catalysts for various types of reactions. Many of these reactions are for renewable energy applications like hydrogen production, turning CO$_2$ into more valuable products, etc. The underlying simulations use quantum mechanical calculations (like you're learning about in your molecular foundations of ChE course), each of which is very slow. There are now many machine learning models (mostly graph neural networks) that can directly predict properties of these simulations. However, these models work much better for some catalyst surfaces and adsorbates than others, and we want your help in finding out how and when to trust these predictions!


`````{note}
**Catalyst dataset summary:**
* 2 CSV files (val_ID_compositions.csv, val_OOD_cat_compositions.csv)
    * val_ID_compositions.csv contains ~25,000 chemical compositions and energies from the OC20 validation split most similar to the training data set. This is probably what you should use for your train/validation splits)
    * val_OOD_cat_Compositions.csv contains ~25,000 compositions and energies from a different distribution of catalysts with similar adsorbates but different chemical composition combinations from the above file. For example, CuPt and CuPd combinations might exist in val_id_compositions.csv, but don't exist in val_OOD_cat_compositions.csv. 
* Each data point contains input features like chemical composition and other information about the surface and adsorbates. It also contains outputs for the actual DFT (simulation) energy, as well as predictions from two different machine learning models. 
`````

## Suggested challenges
* Given the bulk composition (chemical elements and their ratios) and the adsorbate composition (number of each type of atoms), predict the the DFT energies
* Given the bulk composition (chemical elements and their ratios) and the adsorbate composition (number of each type of atoms), predict the difference between one of ML models and the DFT energies. 
* Analyze the most important features from above to figure out which elements are hardest or eastiest to fit. 

````{note}
In each case, you should think critically about the question how you want to set up your train/validation/test splits to simulate the challenge. 
* What do you expect to be correlated?
* How would you know if your model is helpful?
* Is there any way to cheat at the task? 
````

## Dataset/file format details

Two csv files (`val_*.csv`) that together contain ~60,000 DFT energies and predictions using two different types of graph neural networks:       
* `arbitrary number` (disregard the first column)
* `random_id`: a unique identifier for each simulation structure, useful in the mappings below
* `output_DFT_energy`: the ground truth label,  units of [eV]
* `output_gemnet_direct`: energies from a fast direct graph neural network model, units of [eV]
* `output_gemnet_relaxation`: energies from a slower relaxation graph neural network model, units of [eV]
* `input_bulk_mpid` : Materials Project ID of the bulk system used corresponding the the catalyst surface
* `input_bulk_symbols`  Chemical composition of the bulk counterpart
* `input_ads_symbols`  Chemical composition of the adsorbate counterpart
* `input_miller_index`: 3-tuple of integers indicating the Miller indices of the surface
* `input_shift`: c-direction shift used to determine cutoff for the surface (c-direction is following the nomenclature from Pymatgen)
* `input_top`: boolean indicating whether the chosen surface was at the top or bottom of the originally enumerated surface
* `input_class`: integer indicating the class of materials the system's slab is part of, where:
    * 0 - intermetallics
    * 1 - metalloids
    * 2 - non-metals
    * 3 - halides
* `input_anomaly`: integer indicating possible anomalies (based off general heuristics, not to be taken as perfect classifications), where:
    * 0 - no anomaly
    * 1 - adsorbate dissociation
    * 2 - adsorbate desorption
    * 3 - surface reconstruction
    * 4 - incorrect CHCOH placement, appears to be CHCO with a lone, uninteracting, H far off in the unit cell

## Data availability

https://github.com/ulissigroup/F22-06-325/tree/main/f22-06-325/projects/catalysts/data

+++ {"id": "13i7KQ9t-CV8"}

## Background:

+++ {"id": "JkjKcVJ47hSN", "tags": []}

Scalable and cost-effective solutions to renewable energy storage are essential to addressing the world’s rising energy needs while reducing climate change. As illustrated in the figure below, as we increase our reliance on renewable energy sources such as wind and solar, which produce intermittent power, storage is needed to transfer power from times of peak generation to peak demand. This may require the storage of power for hours, days, or months. One solution that offers the potential of scaling to nation-sized grids is the conversion of renewable energy to other fuels, such as hydrogen. To be widely adopted, this process requires cost-effective solutions to running chemical reactions.

An open challenge is finding low-cost catalysts to drive these reactions at high rates. Through the use of quantum mechanical simulations (Density Functional Theory, DFT), new catalyst structures can be tested and evaluated. Unfortunately, the high computational cost of these simulations limits the number of structures that may be tested. The use of AI or machine learning may provide a method to efficiently approximate these calculations; reducing the time required from 24} hours to a second. This capability would transform the search for new catalysts from the present day practice of evaluating O(1,000) of handpicked candidates to the brute force search over millions or even billions of candidates.

As part of OCP, we publicly released the world's largest quantum mechanical simulation dataset -- [OC20](https://github.com/Open-Catalyst-Project/ocp/blob/master/DATASET.md) -- in the Fall of 2020 along with a suite of baselines and evaluation metrics. The creation of the dataset required over 70 million hours of compute. This dataset enables the exploration of techniques that will generalize across different catalyst materials and adsorbates. If successful, models trained on the dataset could enable the computational testing of millions of catalyst materials for a wide variety of chemical reactions. However, techniques that achieve the accuracies required** for practical impact are still beyond reach and remain an open area for research, thus encouraging research in this important area to help in meeting the world's energy needs in the decades ahead.

** The computational catalysis community often aims for an adsorption energy MAE of 0.1-0.2 eV for practical relevance.

![California energy demand](./california_demand.jpg)

The discovery of efficient and economic catalysts (materials) are needed to enable the widespread use of renewable energy technologies. A common approach in discovering high performance catalysts is using molecular simulations. Specifically, each simulation models the interaction of a catalyst surface with molecules that are commonly seen in electrochemical reactions. By predicting these interactions accurately, the catalyst's impact on the overall rate of a chemical reaction may be estimated.

An important quantity in screening catalysts is their adsorption energy for the molecules, referred to as `adsorbates', involved in the reaction of interest. The adsorption energy may be found by simulating the interaction of the adsorbate molecule on the surface of the catalyst to find their resting or relaxed energy, i.e., how tightly the adsorbate binds to the catalyst's surface (visualized below). The rate of the chemical reaction, a value of high practical importance, is then commonly approximated using simple functions of the adsorption energy. The goal of this tutorial specifically and the project overall is to encourage research and benchmark progress towards training ML models to approximate this relaxation.

Specifically, during the course of a relaxation, given an initial set of atoms and their positions, the task is to iteratively estimate atomic forces and update atomic positions until a relaxed state is reached. The energy corresponding to the relaxed state is the structure's 'relaxed energy'. The relaxed energy is very useful for predicting the activity of a catalyst surface!

`````{seealso}
To gain an even better understanding of the Open Catalyst Project and the problems it seeks to address, you can read the following resources:
* Open Catalyst codebase: https://github.com/Open-Catalyst-Project/ocp/
* Open Catalyst webpage: https://opencatalystproject.org/
* [Electrocatalysis white paper](https://arxiv.org/pdf/2010.09435.pdf): C. Lawrence Zitnick, Lowik Chanussot, Abhishek Das, Siddharth Goyal, Javier Heras-Domingo, Caleb Ho, Weihua Hu, Thibaut Lavril, Aini Palizhati, Morgane Riviere, Muhammed Shuaibi, Anuroop Sriram, Kevin Tran, Brandon Wood, Junwoong Yoon, Devi Parikh, Zachary Ulissi: “An Introduction to Electrocatalyst Design using Machine Learning for Renewable Energy Storage”, 2020; arXiv:2010.09435.
* [OC20 dataset paper](https://arxiv.org/pdf/2010.09990.pdf): L. Chanussot, A. Das, S. Goyal, T. Lavril, M. Shuaibi, M. Riviere, K. Tran, J. Heras-Domingo, C. Ho, W. Hu, A. Palizhati, A. Sriram, B. Wood, J. Yoon, D. Parikh, C. L. Zitnick, and Z. Ulissi. The Open Catalyst 2020 (oc20) dataset and community challenges. ACS Catalysis, 2021.
* [Gemnet model:](https://arxiv.org/abs/2106.08903) Johannes Klicpera, Florian Becker, and Stephan Günnemann. Gemnet: Universal directional graph neural networks for molecules, 2021.
`````

+++

## Hints and possible approaches

+++

## Example Model

+++

First, let's load the data using pandas and see what's in it!

```{code-cell} ipython3
import pandas as pd

df = pd.read_csv("data/val_ID_compositions.csv").dropna()
df
```

These formats are not very helpful. Let's try calculating the ratio of each element (one column for each unique element in the dataset)

```{code-cell} ipython3
import re


def composition_to_elements(composition):
    # Helper function to convert a composition like MnAu4 to a
    # dictionary like {'Mn': 1, 'Au': 4}
    elements = {}
    for k, v in re.findall(r"([A-Z][a-z]*)(\d*)", composition):
        if v == "":
            v = 1
        if k in elements:
            elements[k] += int(v)
        else:
            elements[k] = int(v)
    return elements


bulk_element_counts = (
    df["input_bulk_symbols"].apply(composition_to_elements).apply(pd.Series).fillna(0)
)
bulk_element_ratios = bulk_element_counts.div(bulk_element_counts.sum(axis=1), axis=0)
bulk_element_ratios
```

Let's do the same thing for the elements in the adsorbate (the small molecule sitting on the surface

```{code-cell} ipython3
adsorbate_element_counts = (
    df["input_ads_symbols"].apply(composition_to_elements).apply(pd.Series).fillna(0)
)
adsorbate_element_counts
```

As an example, let's construct a full feature matrix by stacking the bulk element ratios and the adsorbate elements counts into one big vector. For the regression target, we'll choose the absolute error between the relaxation machine learning model and the DFT energies. 

Basically, we're trying to see if we can build a model that can predict when the complicated (gemnet) model will fail based on the composition of the adsorbate and the bulk material!

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
import numpy as np

X_train, X_val, y_train, y_val = train_test_split(
    np.hstack((bulk_element_ratios.values, adsorbate_element_counts.values)),
    df["DFT_energy"].values,
)
```

Finally, let's try a random forest model (one of many many possible models that we could try!). We'll fit it and plot the predictions for the validation data.

```{code-cell} ipython3
:tags: []

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# Build a random forest regressor and fit it to the training data
regr = RandomForestRegressor(max_depth=20, random_state=0)
regr.fit(X_train, y_train)

# Plot predictions for val as a kdeplot
sns.kdeplot(
    y_val,
    regr.predict(X_val),
    fill=True,
)

# Add a parity line and x/y axis labels
plt.plot([np.min(y_val), np.max(y_val)], [np.min(y_val), np.max(y_val)], "k--")
plt.axis("square")
plt.xlabel("Actual DFT Energy [eV]")
plt.ylabel("Predicted DFT Energy [eV]")
plt.show()

from sklearn.metrics import mean_absolute_error
print(f'The mean absolute error is {mean_absolute_error(y_val, regr.predict(X_val)):.2f} eV')
```

This model works ok for predicting the DFT energies. There are many things you could try:
* Different models
* Predicting the difference between the expensive ML models and the DFT values (see challenges above)
* Other ways of featurizing the data (especially the bulk compositions)
* Building separate models for each adsorbate type
* ...

+++ {"id": "jXoiLncsU3pe", "jp-MarkdownHeadingCollapsed": true, "tags": []}

````{note}
The below information is not necessary for the project, but helps to explain the original dataset and the type of simulations and energies that are being calculated

````

## Dataset Overview and Details about Calculations



The Open Catalyst 2020 Dataset (OC20) will be used throughout this tutorial. More details can be found [here](https://github.com/Open-Catalyst-Project/ocp/blob/master/DATASET.md) and the corresponding [paper](https://arxiv.org/abs/2010.09990). Data is stored in PyTorch Geometric [Data](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html) objects and stored in LMDB files. For each task we include several sized training splits. Validation/Test splits are broken into several subsplits: In Domain (ID), Out of Domain Adsorbate (OOD-Ads), Out of Domain Catalyast (OOD-Cat) and Out of Domain Adsorbate and Catalyst (OOD-Both). Split sizes are summarized below:

Train
* S2EF - 200k, 2M, 20M, 134M(All)
* IS2RE/IS2RS - 10k, 100k, 460k(All)

Val/Test
* S2EF - ~1M across all subsplits
* IS2RE/IS2RS - ~25k across all splits


![OC20](./OC20.jpg)

+++ {"id": "bSt6h_Q-oqjK"}

## Data Visualization <a name="visual"></a>

```{code-cell} ipython3
:id: HodnfJpE8D0u

import matplotlib

matplotlib.use("Agg")

import os

import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline

params = {
    "axes.labelsize": 14,
    "font.size": 14,
    "font.family": " DejaVu Sans",
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.labelsize": 25,
    "axes.titlesize": 25,
    "text.usetex": False,
    "figure.figsize": [12, 12],
}
matplotlib.rcParams.update(params)


import ase.io
from ase import Atoms
from ase.build import add_adsorbate, fcc100, molecule
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io import extxyz
from ase.io.trajectory import Trajectory
from ase.optimize import LBFGS
from ase.visualize.plot import plot_atoms
from IPython.display import Image
```

+++ {"id": "VRR5C88U8mH1"}

### Understanding the data
We use the Atomic Simulation Environment (ASE) library to interact with our data. This notebook will provide you with some intuition on how atomic data is generated, how the data is structured, how to visualize the data, and the specific properties that are passed on to our models.

+++ {"id": "hEDcCSGD86Hg"}

### Generating sample data

The OC20 dataset was generated using density functional theory (DFT), a quantum chemistry method for modeling atomistic environments. For more details, please see our dataset paper.  In this notebook, we generate sample data in the same format as the OC20 dataset; however, we use a faster method that is less accurate called effective-medium theory (EMT) because our DFT calculations are too computationally expensive to run here. EMT is great for demonstration purposes but not accurate enough for our actual catalysis applications. Below is a structural relaxation of a catalyst system, a propane (C3H8) adsorbate on a copper (Cu) surface. Throughout this tutorial a surface may be referred to as a slab and the combination of an adsorbate and a surface as an adslab.

+++ {"id": "y6Hx8JtXEbW-"}

### Structural relaxations

A structural relaxation or structure optimization is the process of iteratively updating atom positions to find the atom positions that minimize the energy of the structure. Standard optimization methods are used in structural relaxations — below we use the Limited-Memory Broyden–Fletcher–Goldfarb–Shanno (LBFGS) algorithm. The step number, time, energy, and force max are printed at each optimization step. Each step is considered one example because it provides all the information we need to train models for the S2EF task and the entire set of steps is referred to as a trajectory. Visualizing intermediate structures or viewing the entire trajectory can be illuminating to understand what is physically happening and to look for problems in the simulation, especially when we run ML-driven relaxations. Common problems one may look out for - atoms excessively overlapping/colliding with each other and atoms flying off into random directions.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: GEpQz9In9GrX
outputId: 96cd7bc8-2877-4b35-e133-80a10ad81b61
---
###DATA GENERATION - FEEL FREE TO SKIP###

# This cell sets up and runs a structural relaxation
# of a propane (C3H8) adsorbate on a copper (Cu) surface

adslab = fcc100("Cu", size=(3, 3, 3))
adsorbate = molecule("C3H8")
add_adsorbate(adslab, adsorbate, 3, offset=(1, 1))  # adslab = adsorbate + slab

# tag all slab atoms below surface as 0, surface as 1, adsorbate as 2
tags = np.zeros(len(adslab))
tags[18:27] = 1
tags[27:] = 2

adslab.set_tags(tags)

# Fixed atoms are prevented from moving during a structure relaxation.
# We fix all slab atoms beneath the surface.
cons = FixAtoms(indices=[atom.index for atom in adslab if (atom.tag == 0)])
adslab.set_constraint(cons)
adslab.center(vacuum=13.0, axis=2)
adslab.set_pbc(True)
adslab.set_calculator(EMT())

os.makedirs("data", exist_ok=True)

# Define structure optimizer - LBFGS. Run for 100 steps,
# or if the max force on all atoms (fmax) is below 0 ev/A.
# fmax is typically set to 0.01-0.05 eV/A,
# for this demo however we run for the full 100 steps.

dyn = LBFGS(adslab, trajectory="data/toy_c3h8_relax.traj")
dyn.run(fmax=0, steps=100)

traj = ase.io.read("data/toy_c3h8_relax.traj", ":")

# convert traj format to extxyz format (used by OC20 dataset)
columns = ["symbols", "positions", "move_mask", "tags"]
with open("data/toy_c3h8_relax.extxyz", "w") as f:
    extxyz.write_xyz(f, traj, columns=columns)
```

+++ {"id": "Kb77jRtz9fws"}

### Reading a trajectory

```{code-cell} ipython3
:id: mUbvcij59d6I

identifier = "toy_c3h8_relax.extxyz"

# the `index` argument corresponds to what frame of the trajectory to read in, specifiying ":" reads in the full trajectory.
traj = ase.io.read(f"data/{identifier}", index=":")
```

+++ {"id": "b_e6zDVx9pTC"}

### Viewing a trajectory

Below we visualize the initial, middle, and final steps in the structural relaxation trajectory from above. Copper atoms in the surface are colored orange, the propane adsorbate on the surface has grey colored carbon atoms and white colored hydrogen atoms. The adsorbate’s structure changes during the simulation and you can see how it relaxes on the surface. In this case, the relaxation looks normal; however, there can be instances where the adsorbate flies away (desorbs) from the surface or the adsorbate can break apart (dissociation), which are hard to detect without visualization. Additionally, visualizations can be used as a quick sanity check to ensure the initial system is set up correctly and there are no major issues with the simulation.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 680
id: CV5qe6IP9vZg
outputId: 256f97d6-daa7-40fa-ef50-7ba0ca005f9d
---
fig, ax = plt.subplots(1, 3)
labels = ["initial", "middle", "final"]
for i in range(3):
    ax[i].axis("off")
    ax[i].set_title(labels[i])
ase.visualize.plot.plot_atoms(traj[0], ax[0], radii=0.8, rotation=("-75x, 45y, 10z"))
ase.visualize.plot.plot_atoms(traj[50], ax[1], radii=0.8, rotation=("-75x, 45y, 10z"))
ase.visualize.plot.plot_atoms(traj[-1], ax[2], radii=0.8, rotation=("-75x, 45y, 10z"))
```

+++ {"id": "SSR1vQZ1_Ojq"}

### Data contents <a name="contents"></a>

Here we take a closer look at what information is contained within these trajectories.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 9x8w3o17_May
outputId: a6ed3414-774f-4e9c-f211-73379999f6a0
---
i_structure = traj[0]
i_structure
```

+++ {"id": "4CgeShkN_bdJ"}

#### Atomic numbers

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: cMGTQRIz_f2c
outputId: 20442973-b999-4723-ec66-ac169203dfbe
---
numbers = i_structure.get_atomic_numbers()
print(numbers)
```

+++ {"id": "ol4Zi2Gh_qU_"}

#### Atomic symbols

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: cwbxks-i_uVq
outputId: 4960d233-b6c8-42bb-979d-879b6a20cfd4
---
symbols = np.array(i_structure.get_chemical_symbols())
print(symbols)
```

+++ {"id": "x57XplOw_yNw"}

#### Unit cell

The unit cell is the volume containing our system of interest. Express as a 3x3 array representing the directional vectors that make up the volume. Illustrated as the dashed box in the above visuals.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: VWMMzn_i_0vM
outputId: 9fd0343a-9599-4fcb-911d-87ac48974bc0
---
cell = np.array(i_structure.cell)
print(cell)
```

+++ {"id": "XHRbOyaA_97r"}

#### Periodic boundary conditions (PBC)

x,y,z boolean representing whether a unit cell repeats in the corresponding directions. The OC20 dataset sets this to [True, True, True], with a large enough vacuum layer above the surface such that a unit cell does not see itself in the z direction. Although the original structure shown above is what get's passed into our models, the presence of PBC allows it to effectively repeat infinitely in the x and y directions. Below we visualize the same structure with a periodicity of 2 in all directions, what the model may effectively see.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: htvwgCuFAOSB
outputId: 578202d3-f9c5-4857-c2c1-86ee6aaf5aa0
---
pbc = i_structure.pbc
print(pbc)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 400
id: Flzo7aO-RgyA
outputId: 36835a5f-cc91-48d1-ee8b-8fc5112c0cb6
---
fig, ax = plt.subplots(1, 3)
labels = ["initial", "middle", "final"]
for i in range(3):
    ax[i].axis("off")
    ax[i].set_title(labels[i])

ase.visualize.plot.plot_atoms(
    traj[0].repeat((2, 2, 1)), ax[0], radii=0.8, rotation=("-75x, 45y, 10z")
)
ase.visualize.plot.plot_atoms(
    traj[50].repeat((2, 2, 1)), ax[1], radii=0.8, rotation=("-75x, 45y, 10z")
)
ase.visualize.plot.plot_atoms(
    traj[-1].repeat((2, 2, 1)), ax[2], radii=0.8, rotation=("-75x, 45y, 10z")
)
```

+++ {"id": "TWGXcH7AARpy"}

#### Tags

The OC20 dataset consists of systems with several different types of atoms. To help with identifying the index of certain atoms, we tag each atom according to where it is found in the system. There are three categories of atoms: 
- *sub-surface slab atoms*: these are atoms in the bottom layers of the catalyst, furthest away from the adsorbate
- *surface slab atoms*: these are atoms in the top layers of the catalyst, close to where the adsorbate will be placed   
- *adsorbate atoms*: atoms that make up the adsorbate molecule on top of the catalyst.

Tag:

0 - Sub-surface slab atoms

1 - Surface slab atoms

2 - Adsorbate atoms

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: SGZzFhsrB5A2
outputId: 3b2e4e3e-b82f-4e1a-ed88-e53e3040240b
---
tags = i_structure.get_tags()
print(tags)
```

+++ {"id": "0zVhbDL2B8cd"}

#### Fixed atoms constraint

In reality, surfaces contain many, many more atoms beneath what we've illustrated as the surface. At an infinite depth, these subsurface atoms would look just like the bulk structure. We approximate a true surface by fixing the subsurface atoms into their “bulk” locations. This ensures that they cannot move at the “bottom” of the surface. If they could, this would throw off our calculations. Consistent with the above, we fix all atoms with tags=0, and denote them as "fixed". All other atoms are considered "free".

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: FBMUmGrrCD_h
outputId: 4d0aad44-f6bd-491b-d734-5edf5be04031
---
cons = i_structure.constraints[0]
print(cons, "\n")

# indices of fixed atoms
indices = cons.index
print(indices, "\n")

# fixed atoms correspond to tags = 0
print(tags[indices])
```

+++ {"id": "_DHAYeBUCHbN"}

#### Adsorption energy

The energy of the system is one of the properties of interest in the OC20 dataset. It's important to note that absolute energies provide little value to researchers and must be referenced properly to be useful. The OC20 dataset references all it's energies to the bare slab + gas references to arrive at adsorption energies. Adsorption energies are important in studying catalysts and their corresponding reaction rates. In addition to the structure relaxations of the OC20 dataset, bare slab and gas (N2, H2, H2O, CO) relaxations were carried out with DFT in order to calculate adsorption energies.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 5XxYqdM7CMdd
outputId: c2f5ea9c-1614-42ef-fbc0-75fddd7c976f
---
final_structure = traj[-1]
relaxed_energy = final_structure.get_potential_energy()
print(f"Relaxed absolute energy = {relaxed_energy} eV")

# Corresponding raw slab used in original adslab (adsorbate+slab) system.
raw_slab = fcc100("Cu", size=(3, 3, 3))
raw_slab.set_calculator(EMT())
raw_slab_energy = raw_slab.get_potential_energy()
print(f"Raw slab energy = {raw_slab_energy} eV")


adsorbate = Atoms("C3H8").get_chemical_symbols()
# For clarity, we define arbitrary gas reference energies here.
# A more detailed discussion of these calculations can be found in the corresponding paper's SI.
gas_reference_energies = {"H": 0.3, "O": 0.45, "C": 0.35, "N": 0.50}

adsorbate_reference_energy = 0
for ads in adsorbate:
    adsorbate_reference_energy += gas_reference_energies[ads]

print(f"Adsorbate reference energy = {adsorbate_reference_energy} eV\n")

adsorption_energy = relaxed_energy - raw_slab_energy - adsorbate_reference_energy
print(f"Adsorption energy: {adsorption_energy} eV")
```

+++ {"id": "EchgyYxXCUit"}

#### Plot energy profile of toy trajectory

Plotting the energy profile of our trajectory is a good way to ensure nothing strange has occured. We expect to see a decreasing monotonic function. If the energy is consistently increasing or there's multiple large spikes this could be a sign of some issues in the optimization. This is particularly useful for when analyzing ML-driven relaxations and whether they make general physical sense.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 482
id: WffoTL5pCSrg
outputId: 86e7a0fb-7a34-42ee-db58-edd30323eb54
---
energies = [
    image.get_potential_energy() - raw_slab_energy - adsorbate_reference_energy
    for image in traj
]

plt.figure(figsize=(7, 7))
plt.plot(range(len(energies)), energies, lw=3)
plt.xlabel("Step", fontsize=24)
plt.ylabel("Energy, eV", fontsize=24)
```

+++ {"id": "erpOSowgCeuS"}

#### Force

Forces are another important property of the OC20 dataset. Unlike datasets like QM9 which contain only ground state properties, the OC20 dataset contains per-atom forces necessary to carry out atomistic simulations. Physically, forces are the negative gradient of energy w.r.t atomic positions: $F = -\frac{dE}{dx}$. Although not mandatory (depending on the application), maintaining this energy-force consistency is important for models that seek to make predictions on both properties.

The "apply_constraint" argument controls whether to apply system constraints to the forces. In the OC20 dataset, this controls whether to return forces for fixed atoms (apply_constraint=False) or return 0s (apply_constraint=True).

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: NtgLDiT2Cmff
outputId: 61a720bd-4117-4403-eb07-4d49fd5ddc22
---
# Returning forces for all atoms - regardless of whether "fixed" or "free"
i_structure.get_forces(apply_constraint=False)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: QVgvU-OgCqzx
outputId: 1a4bed0b-3554-4b42-b41e-7ca84741d66e
---
# Applying the fixed atoms constraint to the forces
i_structure.get_forces(apply_constraint=True)
```

+++ {"id": "oHIjM6eMwlXY"}

###  Limitations
The OpenCatalyst project is motivated by the problems we face due to climate change, many of which require innovative solutions to reduce energy usage and replace traditional chemical feedstocks with renewable alternatives. For example, one of the most energy intensive chemical processes is the development of new electrochemical catalysts for ammonia fertilizer production that helped to feed the world’s growing population during the 20th century. This is also an illustrative example of possible unintended consequences as advancements in chemistry and materials may be used for numerous purposes. As ammonia fertilization increased in use, its overuse in today’s farming has led to ocean “dead zones” and its production is very carbon intensive. Knowledge and techniques used to create ammonia were also transferred to the creation of explosives during wartime. We hope to steer the use of ML for atomic simulations to societally-beneficial uses by training and testing our approaches on datasets, such as OC20, that were specifically designed to address chemical reactions useful for addressing climate change.4

### Next Steps for State of the Art in Catalyst Simulation Modeling

While progress has been well underway - https://opencatalystproject.org/leaderboard.html, a considerable gap still exists between state-of-the-art models and our target goals. We offer some some general thoughts as to next steps for the readers to ponder on or explore:

* GNN depth has consistenly improved model performance. What limitations to depth are there? How far can we push deeper models for OC20? 
* Our best performing models have little to no physical biases encoded. Can we incorporate such biases to improve our models? Experiments with physically inspired embeddings have had no advantage vs. random initializations, are there better ways to incorporate this information into the models?
* Uncertainty estimation will play an important role in later stages of the project when it comes to large scale screening. How can we get reliable uncertainty estimates from large scale GNNs?
* Are we limited to message-passing GNNs? Can we leverage alternative architectures for similiar or better performance?
* Trajectories are nothing more than sequential data points. How can we use sequential modeling techniques to model the full trajectory?

OC20 is a large and diverse dataset with many splits. For those with limited resources but unsure where to start, we provide some general recommendations:

* The IS2RE-direct task is a great place to start. With the largest training set containing ~460k data points, this task is easily accesible for those with even just a single GPU.
* Those interested in the more general S2EF task don't need to train on the All set to get meaningful performance.
    * Results on the 2M dataset are often sufficient to highlight model improvements.
    * For a fixed compute budget (e.g. fixed number of steps), training on the All set often leads to better performance.
* The S2EF 200k dataset is fairly noisy, trying to find meaningful trends using this dataset can be difficult.
