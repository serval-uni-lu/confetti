# Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification

This repository contains the code to reproduce experiments, evaluations, and figures for our method. Follow the instructions below to replicate the results.

---

## 1. Install requirements

Make sure you are using **Python 3.12+** (preferably in a virtual environment). Then install dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Train the models

To train the baseline models used in the paper, run:

```bash
cd models
python train_models.py
cd ..
```

This will train models for all datasets.

---

## 3. Generate test samples

Next, generate test samples:

```bash
cd benchmark/data
python generate_samples.py
cd ../..
```

---

## 4. Patch TSInterpret

A patched version of TSInterpret is required. Run:

```bash
cd benchmark/generators
python patch_tsinterpret.py
cd ../..
```

This script automatically applies the required modifications.

---

## 5. Generate counterfactuals

Run the benchmark script:

```bash
cd benchmark
python reproduce_benchmark.py
cd ..
```

This will generate counterfactual explanations for all configured methods.

---

## 6. Evaluate results

First, compute the evaluation metrics:

```bash
cd benchmark/evaluations
python evaluations.py
```

Then perform the statistical analysis (Wilcoxon tests):

```bash
python wilcoxon.py
cd ../..
```

---

## 7. Generate figures

Finally, regenerate all figures used in the paper:

```bash
cd figures
python figures.py
cd ..
```

---

## Notes

- If you are running this on a machine with **only one CPU core**, edit the file `evaluations.py` (and similar scripts using `ProcessPoolExecutor`) to set `max_workers=1`. For example:

  ```python
  with ProcessPoolExecutor(max_workers=1) as executor:
  ```

- Make sure all commands are run from the root of the repository, unless stated otherwise.
