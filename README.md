# Hospital Readmission Prediction

**MIS 637 B — Stevens Institute of Technology**  
**Group 7**

| Name | Role |
|---|---|
| Sujay Bhagawan Ghadge | Clustering & Data Analysis |
| Rishi Chhabra | Classification Models & Evaluation |
| Aditya Singh | Data Preprocessing & PCA |
| Purva Sarode | Visualization & Reporting |

---

## Problem Statement

Hospital readmissions within 30 days are one of the biggest challenges in healthcare today, costing the U.S. system approximately **$26 billion annually**. For our MIS 637 project, we wanted to apply data mining techniques to this real-world problem — specifically, predicting whether a diabetic patient will be readmitted to the hospital within 30 days of discharge.

The idea is that if we can flag high-risk patients before they leave, clinicians can intervene early and hopefully prevent that readmission. We used a publicly available dataset from the UCI Machine Learning Repository and built a full pipeline from raw data all the way to model evaluation.

Our three main goals were:
- Use **clustering** to discover hidden patient risk patterns we wouldn't see otherwise
- **Train and compare 6 classification algorithms** to find the best predictor
- **Evaluate every model** thoroughly using Accuracy, Precision, Recall, F1, Confusion Matrix, and ROC Curves

---

## Dataset

We used the **Diabetes 130-US Hospitals (1999–2008)** dataset from the UCI Machine Learning Repository.

| Property | Details |
|---|---|
| Source | UCI Machine Learning Repository |
| Records | 101,766 patient encounters |
| Features | 50 attributes per record |
| Target | `readmitted`: `<30` days / `>30` days / `No` |
| Format | CSV, free and publicly available |
| URL | https://archive.ics.uci.edu/dataset/296 |

The dataset has things like patient demographics, diagnoses codes, lab results, medications, and prior hospital visits. The target we cared about was binarized to **1 = readmitted within 30 days, 0 = everything else**.

One thing we noticed right away: only about **11% of records are positive class** (`<30`). This class imbalance was a big challenge and we had to deal with it using SMOTE during classification.

---

## Project Structure

```
hospital-readmission/
├── data/
│   └── diabetic_data.csv          ← dataset goes here (not versioned)
├── notebooks/
│   ├── 01_preprocessing.ipynb     ← Phase 1
│   ├── 02_clustering.ipynb        ← Phase 2
│   ├── 03_classification.ipynb    ← Phase 3
│   └── 04_evaluation.ipynb        ← Phase 4
├── outputs/
│   ├── figures/                   ← all plots saved here
│   │   ├── class_distribution.png
│   │   ├── pca_scree.png
│   │   ├── kmeans_selection.png
│   │   ├── kmeans_clusters.png
│   │   ├── dendrogram.png
│   │   ├── confusion_matrices.png
│   │   ├── roc_curves.png
│   │   └── model_comparison_bar.png
│   ├── models/                    ← saved model pickles
│   └── model_comparison.csv       ← final results table
├── requirements.txt
└── README.md
```

---

## Setup & How to Run

### 1. Clone the repo

```bash
git clone https://github.com/rchhabra13/hospital-readmission.git
cd hospital-readmission
```

### 2. Install dependencies

```bash
pip3 install -r requirements.txt
```

Dependencies include: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`, `imbalanced-learn`, `scipy`

### 3. Get the dataset

Download `diabetic_data.csv` from https://archive.ics.uci.edu/dataset/296 and place it in the `data/` folder.

```
data/diabetic_data.csv
```

### 4. Run the notebooks in order

You can run them one by one in Jupyter:

```bash
jupyter notebook
```

Or execute all at once from the terminal:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_preprocessing.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_clustering.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_classification.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_evaluation.ipynb
```

Each notebook saves its outputs (figures, arrays, models) so the next one can pick up where it left off.

---

## Methodology — 4 Phase Approach

### Phase 1 — Data Preprocessing (`01_preprocessing.ipynb`)

The raw dataset had a few issues we had to handle:

- **Missing values**: The dataset uses `?` for missing data (not `NaN`). We replaced them and dropped columns with more than 40% missing — specifically `weight`, `payer_code`, and `medical_specialty`.
- **Dropped non-predictive columns**: `encounter_id` and `patient_nbr` are just IDs and don't help a model.
- **Target binarization**: We collapsed the 3-class target (`<30`, `>30`, `No`) into a binary problem: `1` if readmitted within 30 days, `0` otherwise.
- **Categorical encoding**: Used `LabelEncoder` on all object columns.
- **PCA**: Applied `StandardScaler` then PCA keeping 95% of variance, reducing the feature space significantly. This output is used for clustering in Phase 2.

All preprocessed arrays (`X_scaled.npy`, `X_pca.npy`, `y.npy`) are saved to `data/` so we don't have to redo this every time.

---

### Phase 2 — Clustering (`02_clustering.ipynb`)

We wanted to see if there were natural patient groupings in the data before throwing everything at a classifier.

**K-Means Clustering**
- Ran K-Means for K = 2 through 10
- Used the Elbow Method (inertia) and Silhouette Score to pick the best K
- Settled on **K = 3 clusters** based on the plots
- Visualized clusters on the first two PCA components

**Readmission rate per K-Means cluster:**

| Cluster | Readmit Rate (<30d) |
|---|---|
| 0 | Low risk |
| 1 | Medium risk |
| 2 | Highest risk |

(Exact rates shown in the notebook output)

**Hierarchical Clustering**
- Used Ward linkage on a random sample of 500 patients (full dataset too large for a dendrogram)
- Dendrogram shows clear split into 3 natural groups, consistent with K-Means findings
- Also fit `AgglomerativeClustering` on full data for cluster-level readmission rate comparison

---

### Phase 3 — Classification (`03_classification.ipynb`)

We trained 6 classifiers. Because the dataset is heavily imbalanced (~89% negative class), we applied **SMOTE** (Synthetic Minority Over-sampling Technique) on the training set before fitting any model.

**Train/Test split:** 80/20 stratified

| Classifier | Configuration |
|---|---|
| Decision Tree (C4.5) | `criterion='entropy'`, `max_depth=10` |
| Neural Network (BP) | `MLPClassifier`, 2 hidden layers (100, 50), 300 iterations |
| Naïve Bayes | `GaussianNB` |
| KNN | `k=5` |
| Logistic Regression | `max_iter=1000` |
| Random Forest | 100 trees |

All trained models are saved to `outputs/models/all_models.pkl`.

---

### Phase 4 — Model Evaluation (`04_evaluation.ipynb`)

We evaluated every model using the full set of metrics from our proposal.

**Why Recall matters most here:** In a medical context, a **False Negative** (predicting a patient won't be readmitted when they actually will be) is the worst outcome. That patient gets no extra intervention and ends up back in the hospital. So we cared a lot about Recall, not just Accuracy.

---

## Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree (C4.5) | 0.8778 | 0.3277 | 0.1036 | 0.1574 | 0.6233 |
| Neural Network (BP) | 0.7722 | 0.1486 | 0.2256 | 0.1792 | 0.5552 |
| Naïve Bayes | 0.1139 | 0.1101 | **0.9946** | 0.1983 | 0.5886 |
| KNN | 0.6326 | 0.1368 | 0.4398 | 0.2087 | 0.5609 |
| **Logistic Regression** | 0.6487 | 0.1628 | 0.5280 | **0.2488** | **0.6290** |
| Random Forest | **0.8883** | **0.4118** | 0.0322 | 0.0598 | 0.6250 |

### Key Takeaways

**Logistic Regression is our recommended model** for this use case. It achieved the best F1 score (0.2488) and highest ROC-AUC (0.629) while maintaining a reasonable Recall of 52.8%. Given that our priority is minimizing false negatives (missed high-risk patients), F1 and Recall are the metrics that matter most here.

**Random Forest had the highest raw accuracy (88.8%) and precision (41.2%)** but only caught 3.2% of actual readmissions. For a medical intervention tool, this is essentially useless — almost every high-risk patient gets missed.

**Naïve Bayes caught almost every readmission (Recall = 99.5%)** but flagged nearly everyone as high-risk. Its 11.4% accuracy means it's basically predicting the positive class for everyone, which defeats the purpose.

**The core challenge is class imbalance.** Only ~11% of encounters are `<30d` readmissions, even after SMOTE. Future work could explore better resampling strategies, cost-sensitive learning, or ensemble methods tuned for Recall.

---

## Evaluation Metrics Explained

We used the following metrics, as defined in our project proposal:

| Metric | Formula | What it tells us |
|---|---|---|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Overall correct predictions |
| Precision | TP / (TP + FP) | Of patients flagged high-risk, how many actually were |
| Recall | TP / (TP + FN) | Of all true high-risk patients, how many we caught |
| F1 Score | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of Precision and Recall |
| ROC-AUC | Area under TPR vs FPR curve | Model's ability to discriminate between classes |

**In the medical context: False Negatives are the most costly error.** A missed high-risk patient receives no additional monitoring or intervention, which can lead to adverse outcomes.

---

## Figures Generated

All figures are saved to `outputs/figures/`:

| File | Description |
|---|---|
| `class_distribution.png` | Class imbalance visualization |
| `pca_scree.png` | Cumulative explained variance — shows how many PCA components to keep |
| `kmeans_selection.png` | Elbow + Silhouette plots for choosing K |
| `kmeans_clusters.png` | Scatter of clusters on first 2 PCA components |
| `dendrogram.png` | Hierarchical clustering dendrogram (500-patient sample) |
| `confusion_matrices.png` | Side-by-side confusion matrices for all 6 models |
| `roc_curves.png` | ROC curves with AUC for all models on one plot |
| `model_comparison_bar.png` | Bar chart comparing all 5 metrics across all models |

---

## Tools & Technologies

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| pandas | Data loading and manipulation |
| numpy | Array operations |
| scikit-learn | Preprocessing, clustering, classification, evaluation |
| imbalanced-learn | SMOTE for handling class imbalance |
| matplotlib / seaborn | All visualizations |
| scipy | Hierarchical clustering (linkage, dendrogram) |
| Jupyter Notebook | Interactive analysis and documentation |
| Microsoft Excel | Initial data exploration and profiling |

---

## References

- Strack, B. et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates.* BioMed Research International.
- UCI ML Repository: https://archive.ics.uci.edu/dataset/296
- Chawla, N.V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR.

---

*MIS 637 B — Data Mining | Stevens Institute of Technology | Spring 2025*
