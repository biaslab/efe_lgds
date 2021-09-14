## Experiments for the paper "On Epistemics for Expected Free Energy in linear Gaussian state-space models

This repo contains 4 files that recreate the 4 experiments in the paper.

``additive_experiments.jl`` recreate the experiments in sec. 5.1 on pure exploration with additive controls

``multiplicative_experiments.jl`` recreate the experiments in sec. 5.1 on pure exploration with multiplicative controls

The ``efe_experiments_additive.jl`` and ``efe_experiments_multiplicative.jl`` files reproduce the experiments in sec. 5.3

All experiments rely on helper functions defined in ``utils.jl`` and were conducted on Julia v1.6.0 with the environment given in the provided .toml files.
