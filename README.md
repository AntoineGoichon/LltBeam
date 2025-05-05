# Project Setup and Data Generation Guide
---

## Overview

**LltBeam** is a fast and efficient simulator for computing the **dynamic structural response of a wing** subjected to aerodynamic loads. It combines:

- **Lifting-line theory (LLT)** to compute spanwise unsteady aerodynamic forces,
- An **Euler–Bernoulli beam finite element model (FEM)** with a **cantilever configuration** to simulate structural dynamics.

Aerodynamic loads are computed from flight conditions and projected onto the structural model, which is integrated over time using the **Newmark scheme**. The framework is designed for time-domain simulation of flexible wings under a wide range of flight conditions.

--- 

## Installation

### Step 1: Create a Virtual Environment

Create a Python virtual environment and install the following dependencies (NumPy must be < 2.0):

```bash
pip install numpy<2.0 sympy pandas tqdm scipy
```

### Step 2: Install the `LltBeam` Package

Activate your environment, go to the `LltBeam` folder, then run:

```bash
pip install .
```

Or, for development/editing:

```bash
pip install -e .
```

---

## `LltBeam` Package Structure

The package includes the following core files:

- **`llt.py`**: Defines the `Llt` class, used to simulate aerodynamic loads using the lifting-line theory.

- **`beam.py`**: Contains the `AssembleBeamMatrix` class, which assembles Euler–Bernoulli FEM matrices from elementary matrices.

- **`utils.py`**: Provides several utility functions:
  - Computation of elementary matrices used by `AssembleBeamMatrix`
  - Strain-displacement elementary matrix
  - Projection matrix \( P \), used to project aerodynamic loads onto the FEM beam model
  - Modal analysis utilities

- **`main.py`**: Defines the `LltBeam` class, which:
  - Computes aerodynamic loads for a given wing using the lifting-line theory
  - Applies these loads to the corresponding FEM beam model
  - Uses the Newmark scheme for time integration
  - Requires detailed input data (wing parameters, structural matrices, etc.)

---

## Running a Simulation

An example script is provided in `run/run.py`. You can replace the default flight data path with any compatible dataset. The `experimentation_path` specifies the directory where simulation results are saved, including:

- A copy of the input flight data (duplicated from the original `fp` folder),
- The wing and simulation parameters,
- The structural matrices associated with the beam model (e.g., FEM model of the wing).

This example was used to **generate the open-source dataset** available [here](https://doi.org/10.5281/zenodo.15305275). It is based on simulations of **18 flights** generated using a flight simulator, totaling over **26 hours of flight time**. Simulations were originally performed at **500 Hz** and then downsampled to **100 Hz**. The dataset covers a wide range of flight conditions, from **Mach 0.1 to Mach 0.9**, and includes episodes of **atmospheric turbulence**.

The procedure used to extract the wing’s structural and aerodynamic characteristics rely on xfoil and is detailed in:

> Antoine Goichon (2025), *Approches hybrides couplant modèles physiques et apprentissage profond pour la prédiction d’état structurel de structures aéronautiques*, Doctoral Thesis.

---
