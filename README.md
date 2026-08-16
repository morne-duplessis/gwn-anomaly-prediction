# GWN Anomaly Prediction

Source code, experiment configurations, and supplementary material accompanying the paper:

> Du Plessis, M. C., Scharler, J., & Moodley, D.
>
> **Applying Spatial-Temporal Graph Neural Networks for Anomaly Prediction in a Mineral Processing Plant**

## Archived Release

A permanent archived version of this repository is available on Zenodo:

https://doi.org/10.5281/zenodo.21967028

The Zenodo record provides a citable, immutable snapshot of the source code, experiment configurations, and supplementary material associated with the publication.

## Overview

This repository contains the implementation and experimental framework used to evaluate anomaly prediction in industrial time-series data from a mineral processing plant.

The study evaluates Graph WaveNet (GWN) for anomaly prediction and compares its performance against established deep learning baselines, including:

- Long Short-Term Memory (LSTM)
- Temporal Convolutional Networks (TCN)
- Multi-Layer Perceptrons (MLP)

The repository also includes implementations of the weighted loss functions proposed by Du Plessis and Moodley (2025) and evaluates their effectiveness within a spatial-temporal graph learning framework.

### Evaluated Loss Functions

- BCE
- PBCE
- WBCE
- Focal Loss
- WPBCE
- WHinge
- WSqHinge

## Repository Contents

```text
.
├── models/
├── utils/
├── user_preprocessing/
├── supplementary/
│   └── SACAIR2026_GWN_for_anomaly_prediction (supplementary).pdf
├── main.py
├── losses.py
├── config.py
├── requirements.txt
└── README.md