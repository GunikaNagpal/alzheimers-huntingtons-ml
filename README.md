🧬 Alzheimer’s & Huntington’s Disease Classification Pipeline

📌 Overview

This repository implements a bioinformatics + machine learning pipeline to classify brain samples into Alzheimer’s Disease (AD), Huntington’s Disease (HD), and Controls using the GSE33000 microarray gene expression dataset.
The project demonstrates how to go from raw GEO data → preprocessing → feature selection → model training → evaluation with plots and metrics.


📂 Dataset
	•	Source: GSE33000 – NCBI GEO [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE33000]
	•	Samples: 624 (AD: 310, HD: 157, Control: 157)
	•	Features: ~39,280 genes (expression values)


⚙ Pipeline Workflow

	1.	Data Preprocessing
	•	Download/parse GEO dataset with GEOparse
	•	Extract sample IDs + diagnosis (AD / HD / Control)
	•	Build expression matrix (genes × samples)
  
	2.	Feature Selection
	•	Variance filter to remove non-informative genes
	•	ANOVA F-test to keep ~2,000 top genes
  
	3.	Model Training
	•	Logistic Regression (OvR), Random Forest, SVM (RBF)
	•	5-fold stratified cross-validation on train split
  
	4.	Evaluation & Visualization
	•	Confusion Matrix
	•	ROC Curves (AUC ≈ 0.99–1.00)
	•	PCA scatter plot (2D clusters of samples)
	•	Heatmap of top 50 variable genes
	•	Classification Report



📊 Results
	•	Accuracy: ~99% overall
	•	ROC-AUC: ~0.99–1.00 for each class
	•	PCA: Clear separation of AD, HD, Control groups
	•	Heatmap: Distinct gene expression patterns in top 50 genes


🚀 How to Run

1. Clone the repository
git clone https://github.com/GunikaNagpal/alzheimers-huntingtons-ml.git
cd alzheimers-huntingtons-ml

2. Create the conda environment
conda env create -f environment.yml
conda activate ad-hd-classifier

3. Run preprocessing
python -m scripts.parse_labels_and_expr

4. Train models & generate results
python -m scripts.train_model

Outputs will be saved to the outputs/ folder.


📂 Project Structure

alzheimers-huntingtons-ml/
│
├── data/                 # (empty) place GEO dataset here
├── outputs/              # results (plots, reports, CSVs)
├── scripts/              # pipeline scripts
│   ├── parse_labels_and_expr.py
│   └── train_model.py
├── environment.yml       # dependencies
├── README.md             # documentation
└── LICENSE               # MIT license


📜 License

This project is released under the MIT License. Please give credit if you use or adapt this work.


📣 Citation

If you use this repository in your work, please cite:
Gunika Nagpal (2025). Alzheimer’s & Huntington’s Disease Classification Pipeline. GitHub. [https://github.com/GunikaNagpal/alzheimers-huntingtons-ml.git]
