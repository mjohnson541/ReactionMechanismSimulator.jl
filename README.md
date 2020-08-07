# <img align="top" src="https://github.com/ReactionMechanismGenerator/ReactionMechanismSimulator.jl/blob/master/logos/rms-logo-small.png"> RMS - Reaction Mechanism Simulator

[![Build status](https://img.shields.io/travis/ReactionMechanismGenerator/ReactionMechanismSimulator.jl/master.svg)](https://travis-ci.org/ReactionMechanismGenerator/ReactionMechanismSimulator.jl)
[![codecov](https://codecov.io/gh/ReactionMechanismGenerator/ReactionMechanismSimulator.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ReactionMechanismGenerator/ReactionMechanismSimulator.jl)

## Description
RMS is a Julia package designed for simulating and analyzing large chemical reaction mechanisms.  

## Features
Ideal gas and dilute liquid phases.  
Constant T and P and constant V adiabatic ideal gas reactors.  
Constant T and V dilute liquid reactors.  
Diffusion limited rates.
Sensitivity analysis for all reactors.  
Flux diagrams with molecular images (if molecular information is provided).  
Handy plotting and other solution analysis tools.  
Easy to add new features.  

## Installation

RMS currently has dependencies that are only available in python 2.  So currently in order to use RMS it is necessary for your PyCall to reference a python 2 installation.  

This can be done with PyCall uninstalled:  
```
using Pkg
ENV["PYTHON"] = "absolute path to python 2 executable ex:  ~/anaconda2/envs/conda_jl/bin/python"
ENV["CONDA_JL_HOME"] = "absolute path to the python install ex:  ~/anaconda2/envs/conda_jl"
Pkg.add("PyCall")
Pkg.build("PyCall")
```

Once this is done RMS can be installed with:
```
using Pkg
Pkg.add("ReactionMechanismSimulator")
Pkg.build("ReactionMechanismSimulator")

using ReactionMechanismSimulator
```

Detailed instructions and documentation are currently available in the <a href="https://github.com/ReactionMechanismGenerator/ReactionMechanismSimulator.jl/wiki">wiki</a>.

