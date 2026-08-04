# DMML — Data Mining & Machine Learning

Lecture animations for the Master's-level DMML course, built with [Manim Community](https://www.manim.community/).

---

## Week 01 — Setup & EDA

### Exploratory Data Analysis (Column Profiling + Distributions + Correlation Heatmap)
![EDA](w01_setup_eda/EDA.gif)

### Data Cleaning (Missing Values + Encoding + Normalisation)
![DataCleaning](w01_setup_eda/DataCleaning.gif)

### Descriptive Statistics (Central Tendency + Box Plot + Skewness)
![StatsOverview](w01_setup_eda/StatsOverview.gif)

---

## Week 02 — Time Series & ARIMA

### Time Series Decomposition
![Decomposition](w02_time_series_arima/Decomposition.gif)

### Stationarity & Differencing
![Stationarity](w02_time_series_arima/Stationarity.gif)

### AR(1) Process
![AR1](w02_time_series_arima/AR1.gif)

### Forecast Cone
![ForecastCone](w02_time_series_arima/ForecastCone.gif)

---

## Week 03 — Regression & Classification

### Bias–Variance Tradeoff
![BiasVariance](w03_regression_classification/BiasVariance.gif)

### Decision Boundary (Logistic Regression vs kNN)
![DecisionBoundary](w03_regression_classification/DecisionBoundary.gif)

---

## Week 04 — SVMs & Model Tuning

### SVM Max-Margin
![SVMMargin](w04_svm_tuning/SVMMargin.gif)

### Kernel Trick
![KernelTrick](w04_svm_tuning/KernelTrick.gif)

### Cross-Validation
![CrossValidation](w04_svm_tuning/CrossValidation.gif)

---

## Week 05 — Trees & Ensembles

### Decision Tree (Splits & Depth)
![DecisionTree](w05_trees_ensembles/DecisionTree.gif)

### Bagging & Bootstrap
![Bagging](w05_trees_ensembles/Bagging.gif)

### Feature Importance (Permutation)
![FeatureImportance](w05_trees_ensembles/FeatureImportance.gif)

---

## Week 06 — Boosting

### Gradient Boosting
![GradientBoosting](w06_boosting_shap/GradientBoosting.gif)

### SHAP Values
![SHAPValues](w06_boosting_shap/SHAPValues.gif)

---

## Week 07 — Clustering & Dimensionality Reduction

### K-Means (Lloyd's Algorithm + Elbow Method)
![KMeans](w07_clustering_pca/KMeans.gif)

### PCA (Principal Components + Projection + Explained Variance)
![PCA](w07_clustering_pca/PCA.gif)

### DBSCAN (ε-ball Expansion + Core/Border/Noise)
![DBSCAN](w07_clustering_pca/DBSCAN.gif)

---

## Week 08 — Neural Networks Intro

### MLP Forward Pass
![MLPForward](w08_neural_networks_intro/MLPForward.gif)

### Gradient Descent
![GradientDescent](w08_neural_networks_intro/GradientDescent.gif)

### Activation Functions & Vanishing Gradient
![ActivationFunctions](w08_neural_networks_intro/ActivationFunctions.gif)

---

## Week 09 — Training Techniques & CNNs

### Convolution (Sliding Filter + Feature Maps + Max Pooling)
![Convolution](w09_training_cnns/Convolution.gif)

### Dropout & Batch Normalization
![DropoutBatchNorm](w09_training_cnns/DropoutBatchNorm.gif)

### CNN Architecture (LeNet Pipeline + Hierarchical Features + Transfer Learning)
![CNNArchitecture](w09_training_cnns/CNNArchitecture.gif)

---

## Week 10 — Sequences & Transformers

### RNN vs LSTM (Unrolled RNN + Vanishing Gradient + LSTM Gates)
![RNNvsLSTM](w10_sequences_transformers/RNNvsLSTM.gif)

### Attention Mechanism (Seq2seq Bottleneck + Attention Weights + Self-Attention Matrix)
![Attention](w10_sequences_transformers/Attention.gif)

### Transformer Architecture (Positional Encoding + Encoder Layer + Classification Head)
![Transformer](w10_sequences_transformers/Transformer.gif)

### Practical Fine-Tuning (Pre-train Cost + Tokenization + Strategies + LR Sensitivity)
![FineTuning](w10_sequences_transformers/FineTuning.gif)

---

## Running the animations yourself

```bash
# clone and create the environment
git clone <repo-url>
cd dmml
python -m venv env && source env/bin/activate
pip install manim

# render any scene (example)
cd animations/w02_time_series_arima
../../env/bin/manim -pql decomposition.py Decomposition
```
