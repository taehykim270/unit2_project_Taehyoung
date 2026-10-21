# Reproducing Wainwright (2009): Sharp Thresholds for Lasso Support Recovery

This project reproduces the simulation study (Figs. 1–3) of
M. J. Wainwright (2009), *Sharp Thresholds for High-Dimensional and Noisy Sparsity
Recovery Using ℓ1-Constrained Quadratic Programming (Lasso)*, IEEE Trans. Inf. Theory 55(5).
The paper shows that the Lasso recovers the exact signed support of a sparse vector
with probability → 1 when the rescaled sample size θ = n / (2k log(p − k)) exceeds a
threshold (θ* = 1 for i.i.d. Gaussian designs), and fails with probability → 1 below it.
The full simulation design is in [ADEMP.md](ADEMP.md).

## Setup

Requires Python 3.10+ and `make`. From the project root:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Activate the environment (`source venv/bin/activate`) in every new terminal before
running any `make` target.


# Simulation setting
venv
source venv bin activate

# Run simulation
How to use make simulate
Estimated run time

# Main discoveries