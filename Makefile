PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help venv data prep features base pso eval symptom lesion all clean

help:
	@echo "Tri-Net MPOX — targets:"
	@echo "  make venv      create venv + install requirements"
	@echo "  make data      download datasets from Kaggle (needs ~/.kaggle token)"
	@echo "  make prep      build 14-class split + clean symptom CSV"
	@echo "  make features  cache frozen-backbone features (.npy)"
	@echo "  make base      train the 3 classification heads"
	@echo "  make pso       PSO ensemble weight optimization"
	@echo "  make eval      figures + tables (confusion/ROC/Kappa/McNemar)"
	@echo "  make symptom   symptom CNN + baselines + 'sum' ablation"
	@echo "  make lesion    prep->features->base->pso->eval (image pipeline)"
	@echo "  make all       the full pipeline end to end"

venv:
	python3 -m venv .venv && $(PIP) install --upgrade pip -q && $(PIP) install -r requirements.txt

data:
	$(PY) -m src.data.download

prep:
	$(PY) -m src.data.prepare_lesion
	$(PY) -m src.data.prepare_symptom

features:
	$(PY) -m src.models.extract_features

base:
	$(PY) -m src.models.train_base

pso:
	$(PY) -m src.models.pso_ensemble

eval:
	$(PY) -m src.eval.run_all

symptom:
	$(PY) -m src.models.symptom_model

lesion: prep features base pso eval

all: data prep features base pso eval symptom

clean:
	rm -rf results/figures/* results/tables/* results/models/*
