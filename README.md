Minimal PyTorch Library for Natural Evolution Strategies
========================================================
<p align="middle">
    <img src="/gifs/lunarlander.gif" width="100%" />
</p>


# Requirements

```
pipcs
numpy
torch
gym
mpi4py
```

To install `mpi4py` you need `MPI` installed in the system.
Check: https://mpi4py.readthedocs.io/en/stable/install.html

A Dockerfile is provided for convenience.

# Installation

```bash
pip install nes-torch --user
```

# Documentation
https://nestorch.readthedocs.io/

# Usage
See https://github.com/goktug97/nes-torch/blob/master/example.py
https://github.com/goktug97/nes-torch/blob/master/minimal_example.py

Check https://github.com/goktug97/pipcs to understand the configuration system.

Check documentation or https://github.com/goktug97/nes-torch/blob/master/nes/config.py for parameters.

You can run the example with
```bash
python example.py
```
or in parallel for faster training.
```bash
mpirun -np 2 python example.py
```
