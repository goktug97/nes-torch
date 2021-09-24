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
```

## Optional
```
gym # For gym examples
mpi4py # For parallelization
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
See https://github.com/goktug97/nes-torch/blob/master/examples

Check https://github.com/goktug97/pipcs to understand the configuration system.

Check documentation or https://github.com/goktug97/nes-torch/blob/master/nes/config.py for parameters.

You can run the example with
```bash
PYTHONPATH="$(pwd):$PYTHONPATH" mpirun --np 2 python examples/example.py
```
or in parallel for faster training.
```bash
PYTHONPATH="$(pwd):$PYTHONPATH" mpirun --np 2 python examples/example.py
```
