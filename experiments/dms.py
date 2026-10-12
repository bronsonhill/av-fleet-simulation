"""
Dymnamic merge sensing (DMS) grid sweep experiment.

- grid search on degree of DMS (minimum merge gap between AVs) from the HDV default value (no DMS) to a shorter gap.
- fixed:
    - density: at max. calibrated flow reasoned fixed value - between jam and free flow
    - AV share chosen by MPR predictions from literature. 50/50 AV fleet split to avoid confounding factors.
    - HDV minimum merge gap: out of scope, could be explored in future work, but chose reasoned default inspired by typical values in literature.
- sensitivity analysis on density: - how sensitive is flow and velocity to DMS degree at different densities? ± 2 vehicles/km/lane from the calibrated density.

"""
