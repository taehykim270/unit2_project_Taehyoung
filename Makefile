# Reproducing Wainwright (2009) - simulation pipeline
#   make reproduce everything from scratch: setup -> clean -> test -> all
#   make setup     create venv/ and install requirements.txt
#   make all       run simulation -> summaries -> figures
#   make simulate  run simulations and save raw results
#   make analyze   summarize raw results
#   make figures   make all figures
#   make test      run test suite
#   make clean     remove generated files

# Use the project venv if it exists, otherwise whatever python3 is on PATH
PYTHON ?= $(if $(wildcard venv/bin/python),venv/bin/python,python3)
# Interpreter used once to create the venv
BOOTSTRAP_PYTHON ?= python3

RAW      := results/raw/simulation_raw.csv
SUMMARY  := results/summary.csv
FIG1     := results/figures/fig1_success_vs_n_and_theta.png

.PHONY: reproduce setup all simulate analyze figures test clean

# One command for a full from-scratch reproduction. Sub-makes run in sequence, so the steps
# after setup use the venv it creates (even on a fresh clone without venv/)
reproduce:
	$(MAKE) setup
	$(MAKE) clean
	$(MAKE) test
	$(MAKE) all

# Create the virtual environment (if missing) and install dependencies
setup:
	test -d venv || $(BOOTSTRAP_PYTHON) -m venv venv
	venv/bin/python -m pip install --upgrade pip
	venv/bin/python -m pip install -r requirements.txt

# Run complete simulation pipeline and generate all outputs
all: figures

# Run the simulation and save raw results
simulate: $(RAW)

$(RAW): src/simulation.py src/dgps.py src/methods.py
	$(PYTHON) -m src.simulation

# Based on the raw results, make summary statistics
analyze: $(SUMMARY)

$(SUMMARY): $(RAW) src/metrics.py data/paper_values.csv
	$(PYTHON) -m src.metrics

# Make the figures from the summary statistics
figures: $(FIG1)

$(FIG1): $(SUMMARY) src/figures.py
	$(PYTHON) -m src.figures

# Run tests
test:
	$(PYTHON) -m pytest

# Remove the generated files (keeps the .gitkeep placeholders)
clean:
	rm -f results/raw/*.csv results/raw/*.json
	rm -f results/summary.csv results/thresholds.csv results/collapse.csv results/paper_comparison.csv
	rm -f results/theory_bounds.csv
	rm -f results/figures/*.png results/figures/*.pdf
	rm -rf .pytest_cache
	find . -name "__pycache__" -type d -not -path "./venv/*" -prune -exec rm -rf {} +
