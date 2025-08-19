<img width="1951" height="1051" alt="Asset 18" src="https://github.com/user-attachments/assets/165c5090-3d3a-4375-9aad-ed05859b34d9" />
# Supplementary Materials

This repository provides supplementary materials for the article **"…"**.  
The content is organized into three main sections: **Code**, **Data**, and **Output**.  

---

## 1. Code

This section contains all scripts and notebooks used for model training, evaluation, and visualization.  

### 1.1 Modules
The `Modules/` directory is divided into the following parts:

- **HelperFunctions.py**  
  Utility functions for data aggregation and visualization, shared across models.  

- **FingerPrints/**  
  Functions for generating the custom fingerprints used in this study.  

- **LinearModels/**  
  Includes:  
  - Data pipeline for linear models  
  - Nested cross-validation  
  - Evaluation tools  

- **NeuralNetworks/**  
  Contains two subdirectories:  
  - **BCELoss/** – Models trained with Binary Cross-Entropy loss  
  - **MRLoss/** – Models trained with Margin Ranking loss  

  Each subdirectory includes:  
  - Model implementation  
  - Data pipeline  
  - Cross-validation setup  

---

### 1.2 Jupyter Notebooks
Six notebooks illustrate model implementations, fingerprint workflows, and result analysis:

- **LinearRegression.ipynb** – Linear regression training and statistics aggregation  
- **LogisticRegression.ipynb** – Logistic regression training and statistics aggregation  
- **NNBCE.ipynb** – Neural network with BCE loss (implementation + statistics)  
- **NNMR.ipynb** – Neural network with MR loss (implementation + statistics)  
- **FingerPrintsOverview.ipynb** – Examples of custom fingerprint workflows  
- **Vis&Stats.ipynb** – Aggregated results, visualizations, and statistical testing  

---

## 2. Data

This section contains the datasets used in the study:  

- **IDB.csv** – Initial Database (IDB)  
- **ILD.csv** – Independent Literature Data (ILD)  

---

## 3. Output

This section contains results and aggregated data from model experiments:  

- **Accumulators_IDB/**  
  Nested CV error accumulations on the IDB dataset (used for error distribution plots).  

- **Accumulators/**  
  Error accumulations for models trained on ILD (second part of the study).  

- **IDB_results_p1/** and **IDB_results_p2/**  
  Cross-validation fold results for Part 1 and Part 2.  

- **ILD_results_p1/**  
  ILD results obtained after training on IDB (Part 1).  

---

