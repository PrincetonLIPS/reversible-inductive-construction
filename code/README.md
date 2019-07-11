# Discrete Object Generation with Reversible Inductive Construction

This folder contains the code to train and run the models.


## Requirements and installation

The training code requires `rdkit`, `networkx`, `tqdm`, `pytorch >= 1.1.0`, `tb-nightly` and the standard
numerical python packages. There are also optional compiled components that we make use
of to speed up the models (they improve the training performance by a factor of 100x-1000x).
To install those, make sure that you have `cmake`, `ninja` and `boost` (all can be conda installed),
as well as the CUDA toolkit and compiler, and run:
```
python setup.py build_ext --inplace
```

## Molecule Models

The main code for the molecule models can be found under the `induc_gen.molecule_models` module.
A simple molecule model can be trained by using:
```
python -m induc_gen.molecule_models.train_joint
```

In general, we strongly recommend using native extensions to significantly accelerate the processing.
We note that the corruption process can be fairly CPU intensive, and so we recommend using a large
number of CPU cores and workers. Alternatively, it is possible to pre-generate corruption datasets
by using the `induc_gen.corruption_dataset` module.

To run the trained models, the `induc_gen.deploy.chain` module can be used. This module will
sample a single chain and save the resulting chain at the specified location.

## Laman Models

The main code for the laman models can be found under the `induc_gen.laman` module.
A simple model can be trained by using:
```
python -m induc_gen.laman.train_joint
```

Note that the training can be quite gpu-memory intensive at large batch sizes.
The provided dataset is only a sample (and only has 1000 observations). It is
possible to generate new datasets by running the following script:
```
python -m induc_gen.laman.data_gen
```
Please see the script for different configuration options.

To run trained models, the `induc_gen.laman.deploy` module can be used.
