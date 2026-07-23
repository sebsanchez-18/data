# GoDurham – Bayesian Cleaning Priority Modeling

## Overview

This folder contains the modeling and analytical workflow developed for the GoDurham Bus Stop Cleaning Priority Project. The project replaces a heuristic cleaning schedule with a data-driven framework by combining bus stop characteristics, ridership, complaint history, demographic information, and spatial analysis.

The workflow consists of two notebooks that should be read in sequence.

---

## Notebook 1 — Bayesian Modeling

**Bayesian_Modeling.ipynb**

This notebook develops a Bayesian hierarchical Zero-Inflated Negative Binomial (ZINB) model to estimate the expected cleaning need for each bus stop.

The notebook includes:

- Data preparation and feature engineering
- Model specification
- Bayesian model fitting using PyMC
- Model diagnostics and convergence assessment
- Posterior prediction of cleaning need

Outputs include:

- Posterior parameter summaries
- Trace plots
- Estimated census tract effects
- Predicted cleaning need

---

## Notebook 2 — Results & Spatial Analysis

**Results_and_Spatial_Analysis.ipynb**

This notebook interprets and visualizes the model outputs.

The analysis includes:

- Infrastructure inventory summaries
- Equity analysis
- Spatial distribution of predicted cleaning need
- Proposed cleaning priority tiers
- Comparison between current and proposed cleaning priorities
- Identification of the highest- and lowest-priority bus stops
---

## Repository Workflow

The notebooks are intended to be read in the following order:

1. Bayesian Modeling
2. Results & Spatial Analysis

---

## Software

- Python
- PyMC
- ArviZ
- pandas
- NumPy
- GeoPandas
- Matplotlib

---

## Data Availability

The original datasets used in this project were developed through a collaboration with the City of Durham and are not included in this public repository.

The notebooks document the complete analytical workflow. Users may substitute equivalent datasets with the same structure to reproduce the analysis.