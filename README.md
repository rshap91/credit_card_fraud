# Identifying Credit Card Fraud

This project is an exploration of the [kaggle credit card fraud detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
dataset. The project was a great opportunity to learn about outlier/anomaly detection
techniques as well as to build models dealing with heavily imbalanced classes.

The dataset provide was very clean so I got to focus more on implementing
streamlined model building pipelines and learning about various anomaly detection
techniques.


### Overview

I took three different approaches to working with this dataset.

    1. Use classifiers to model the difference between real and fraudulent charges.
        - Logistic Regression
        - NaiveBayes
        - Tree Ensembles
        - KNN
        - Try Over and Under Sampling
        - Data Transformations such as scaling and deskewing
    2. Use statistical and un-supervised anomaly detection techniques to try to identify
      core boundary of real charges and identify anything outside this boundary as fraudulent.
        - Robust Covariance estimates
        - Local Outlier Factor
        - Isolation Forests
        - One Class SVM
        - Clustering
          - DBSCAN, Hierarchical, Model-based Bayesian Clustering
    3. Combining steps 1 and 2.
        - Use the outputs from the anomaly detection techniques as additional features
        in the classification models.

### Results

  TBD
