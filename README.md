# TFE-Transfo-ELM-LSTM

This repository contains the MATLAB and Python codes associated with the final-year project:

**Detection and classification of anomalies in an oil-immersed power transformer using ELM and LSTM**

## MATLAB codes

### `ThermalIEC_3PH_OFAF.m`
Three-phase thermal model inspired by IEC 60076-7 for an OFAF oil-immersed power transformer.

Main outputs:
- top-oil temperature
- winding temperatures of phases A, B and C

Main inputs:
- RMS phase currents
- ambient temperature
- pump degradation factor
- fan degradation factor

This file is used as the thermal core for anomaly simulation.

### `simulation_generation_dataset.m`
Automated MATLAB/Simulink script for generating the final raw dataset.

Main tasks:
- simulate the 7 operating classes
- vary load, temperature, power factor and fault parameters
- export SCADA signals to CSV
- generate `index.csv` and `scenarios.mat`

Expected output:
- 504 raw CSV files
- indexed scenario table
- full scenario metadata

## Notes
- The Simulink model name used in the script is `FUNA_ACTUALISE`
- The dataset generation process is based on physically coherent thermal and electrical behavior
- This repository is part of an academic final-year project in electrical engineering
