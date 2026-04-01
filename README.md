# TFE-Transfo-ELM-LSTM

This repository contains the MATLAB and Python codes associated with the final-year project:

**Detection and classification of anomalies in an oil-immersed power transformer using ELM and LSTM**

---

## Project overview

The project combines:
- physics-informed thermal modeling of an OFAF oil-immersed power transformer,
- automated dataset generation from Simulink simulations,
- feature-based preprocessing for ELM,
- sequence-based preprocessing for LSTM,
- training and evaluation of ELM and LSTM models for anomaly detection and classification.

The targeted classes are:

- **CL0**: Normal
- **CL1**: Overload
- **CL2**: Unbalance Phase A
- **CL3**: Unbalance Phase B
- **CL4**: Unbalance Phase C
- **CL5**: Fan fault
- **CL6**: Pump fault

---

## Repository structure

```text
TFE-Transfo-ELM-LSTM/
│
├── README.md
├── matlab/
│   ├── ThermalIEC_3PH_OFAF.m
│   └── simulation_generation_dataset.m
│
├── python/
│   ├── preprocessing_elm.py
│   ├── train_elm.py
│   ├── preprocessing_lstm.py
│   └── train_lstm.py
│
└── models/
    └── best_lstm.keras   (generated after training)
