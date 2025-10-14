Alzheimer's & Huntington's Gene Expression Classifier
A machine learning pipeline for classifying Alzheimer's Disease (AD), Huntington's Disease (HD), and Control samples using gene expression data (GSE33000 dataset).

Project Overview:
- Data parsing from GEO dataset
- Preprocessing (normalization, cleaning)
- Feature selection & dimensionality reduction (PCA, variance filtering)
- Classification with Logistic Regression
- Model evaluation using Confusion Matrix, ROC-AUC, Accuracy
  
Dataset:
- Source: GSE33000 - Gene Expression Omnibus (GEO)
- Samples: 624 (AD: 310, HD: 157, Control: 157)
- Features: 39,280 genes
  
Methods:
1. Data Preprocessing (parse GEO files, extract labels + expression matrix)
2. Feature Selection (variance filter, PCA)
3. Model Training (Logistic Regression, One-vs-Rest)
4. Evaluation (Confusion Matrix, ROC, Heatmaps)
   
Results:
- High accuracy (~99%)
- Confusion Matrix, PCA 2D Plot, Heatmap (Top 50 Genes), ROC Curve
  
How to Run:
git clone https://github.com/GunikaNagpal/alzheimers-huntingtons-ml.git

Create environment:
conda env create -f environment.yml
conda activate ad-hd-classifier

Run preprocessing:
python -m scripts.parse_labels_and_expr

Train & evaluate models:
python -m scripts.train_model

Project Structure:
- data/ : GEO data
- outputs/ : results, plots
- scripts/ : parsing, training, evaluation scripts
- environment.yml : dependencies
- README.md : documentation
