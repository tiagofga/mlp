# Experiments Log Template

Use this file as a standard format for academic experiment tracking. Copy the template section for each run.

## How to use

1. Duplicate the `Experiment Entry` block.
2. Fill all fields before and after running.
3. Keep one entry per experiment ID.
4. Commit this file with code changes when possible.

---

## Experiment Entry

### Metadata

- Experiment ID: `EXP-YYYYMMDD-XX`
- Date: `YYYY-MM-DD`
- Author:
- Branch/Commit:
- Objective:
- Hypothesis:

### Environment

- OS:
- Compiler (`g++`/`clang++` + version):
- CMake version:
- Backend: `CPU` | `OpenMP` | `CUDA` | `OpenMP+CUDA`
- Build command:
- Run command:
- Hardware (CPU/GPU):
- Random seed:

### Model and Training Setup

- Dataset:
- Train/Validation/Test split:
- Input features:
- Output targets:
- Architecture:
- Activation(s):
- Loss:
- Optimizer:
- Learning rate:
- Epochs:
- Batch size:
- Regularization:

### Results

- Final train loss:
- Final validation loss:
- Final test loss:
- Metrics (accuracy/F1/etc.):
- Runtime:
- Notes on convergence:

### Analysis

- Did the hypothesis hold?
- Main failure modes:
- What changed vs previous experiment?
- Threats to validity:

### Next Actions

1. 
2. 
3. 

---

## Example (short)

- Experiment ID: `EXP-20260311-01`
- Objective: Improve XOR convergence speed with larger hidden layer.
- Changes: `Dense(2,8)` -> `Dense(2,16)`, same optimizer and epochs.
- Result: Lower loss after 1000 epochs, similar final predictions.
- Next: test lower learning rates (`0.1`, `0.05`).
