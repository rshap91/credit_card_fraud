# Outlier Detection

Taking two different approaches.

    1. Try to model the difference between real and fraudulent charges.
        - Classifiers like
          - Logistic Regression
          - NaiveBayes
          - Tree Ensembles
          - KNN
        - Try Over and Under Sampling
        - Data Transformations such as deskewing
    2. Try to identify core boundary of real charges and identify anything outside this boundary as fraudulent.
        - Covariance estimates,
        - Local Outlier Factor,
        - Clustering
          - KMeans, DBSCAN, Hierarchical, Model-based Bayesian Clustering
        - One Class SVM
