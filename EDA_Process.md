
## Exploring

Checkout the data, looking for cleaning opportunities while you explore.
Things to look for include:

  - General Distribution of Features
  - Skew if any
  - Outliers
  - Nulls
  - Correlations/Collinearity
  - Target Distribution


## Cleaning

  1. Check and keep track of column dtypes
  2. Check for Anomalies?
  3. Fill Missing Values
  4. Identify Categoricals and Encode
  5. Remove Columns W STD of 0


## Model Prep

  - Look for correlations
    - Pairplot to visualize trends
    - Violin plot with target in the x-axis to visualize distribution by class (look for separation of distributions)
  - Feature Engineering
  - Feature Selection
  - Imbalanced Classes:
    - Can I sequentially train on proportioned samples?
    - Undersample/Oversample Majority?


## Modeling

  - Ensembling is cool, but only thing that worked here was majority vote
  - Instead of Stacking at the end, add all the preds in as features in the
    beginning and then run through feature selection (like what you did with NaiveBayes preds)
  - What I didn't get to try was fitting estimators to _specialize_ in individual classes.
