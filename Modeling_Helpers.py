# -- coding utf8 --
# Title: Modeling Helpers
# Author: Rick Shapiro
"""
Functions to 
    - Calculate F1 scores from Precisions and Recalls.
    - Get various types of samples of a dataset\
    - Plot PR curve at various thresholds
    - Grid Search a classifer with given params
    - Grid search numerous classifiers with various sets of params
""";



import numpy as np
import pandas as pd
import Clean_Function_Helpers as cfh
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_val_score, GridSearchCV



def calc_f1(recalls, precisions):
    return (2*recalls*precisions)/(recalls + precisions)


def get_sample(df, n=None, frac=None, n_oversample=False, n_undersample=False, seed=None):
    """
    Generate Train and Test samples of specified size with the option
    to oversample or undersample.
    """
    if n_oversample:
        # split out classes
        c0 = df[df.Class==0].sample(n = n_oversample//2, random_state=seed)
        # sample half of the minority class
        c1 = df[df.Class==1].sample(frac=0.5, random_state=seed)
        # sample the minority class W REPLACEMENT
        c1 = c1.sample(n_oversample//2, replace=True, random_state=seed)
        # join them together
        train = pd.concat([c0,c1])
        test = df[~df.index.isin(train.index)]
        print('Train Distribution of Target Class')
        print(train.Class.value_counts())
        return train, test
    elif n_undersample:
        # take `n_undersample` observations from each class
        idx0 = df[df.Class==0].sample(n = n_undersample, random_state=seed).index.tolist()
        idx1 = df[df.Class==1].sample(n = n_undersample, random_state=seed).index.tolist()
        tidx = idx0 + idx1
        train = df.loc[tidx].copy() 
        test = df[~df.index.isin(tidx)]
        print('Train Distribution of Target Class')
        print(train.Class.value_counts())
        return train, test
    else:
        train = df.sample(n=n, frac=frac, random_state=seed)
        test = df[~df.index.isin(train.index)]
        print('Train Distribution of Target Class')
        print(train.Class.value_counts())
        return train, test

def pr_curve(y_actual, y_pred, digit_prec=2):
    '''
    PLOTS THE PRECISION VS RECALL OF ESTIMATOR
    OVER A RANGE OF THRESHOLDS

    Y_PRED MUST BE PREDICTED PROBABILITY VECTOR OF ONE CLASS (EG PROBS[:,1])
    MUST ALL BE OF TYPE INT OR FLOAT

    RETURNS: recalls, precisions, thresholds

    '''
    threshvec = np.unique(np.round(y_pred,digit_prec))
    numthresh = len(threshvec)
    tpvec = np.zeros(numthresh)
    fpvec = np.zeros(numthresh)
    fnvec = np.zeros(numthresh)

    for i in range(numthresh):
        thresh = threshvec[i]
        tpvec[i] = sum(y_actual[y_pred>=thresh])
        fpvec[i] = sum(1-y_actual[y_pred>=thresh])
        fnvec[i] = sum(y_actual[y_pred<thresh])
    recallvec = tpvec/(tpvec + fnvec)
    precisionvec = tpvec/(tpvec + fpvec)
    plt.plot(precisionvec,recallvec)
    plt.axis([0, 1, 0, 1])
    plt.xlabel("precision")
    plt.ylabel("recall")
    return (recallvec, precisionvec, threshvec)
    
    
def evaluate_classifier(train_data, test_data, clf, params, cv=4, scoring='f1', drop_cols=('Time', 'Class')):
    '''
    Takes grid searches parameters for a passed classifier and evaluates 
    on the test set.
    '''
    x = train_data.drop(drop_cols, axis=1)
    y = train_data.Class

    # grid search parameters
    grid = GridSearchCV(clf, params, cv=cv, scoring=scoring, n_jobs=-1)
    grid.fit(x,y)
    print('Best Params:', grid.best_params_)
    print('Best Score:', grid.best_score_)

    # get best estimator and scores
    mod = grid.best_estimator_
    # run predictions on test set
    pred_proba = mod.predict_proba(test_data[x.columns])
    preds = mod.predict(test_data[x.columns])
    
    # plot pr curve
    r,p,t = pr_curve(test_data.Class, pred_proba[:,1]);
    
    # find threshold that maximizes f1 score
    idx = np.argmax(calc_f1(r[:-1],p[:-1]))
    stats = {
        'auc': grid.best_score_,
        'f1': calc_f1(r[idx], p[idx]),
        'recall': r[idx],
        'precision': p[idx],
        'threshold': t[idx]
    }
    print('F1:', calc_f1(r[idx], p[idx]))
    print('Recall:', r[idx])
    print('Precision:', p[idx])
    print('Threshold:', t[idx])
    return mod, x, y, stats


def test_all_models(transforms_labels, classifier_params, 
                    sampling_params=None, drop_cols=None, S=5, frac_add_random_noise=0, seed=1111):
    
    sampling_params = sampling_params or {'frac':0.2}
    assert not all([k in sampling_params for k in ['n_undersample', 'n_oversample']]), \
    "Can only specify one of 'n_undersample' or 'n_oversample' for sampling params."
    assert 0 <= frac_add_random_noise <=1, "random noise factor should be between 0 and 1."
    
    results = {}
    for data_transformed, transform_label in transforms_labels:
        for clf, params in classifier_params:
            model_name = str(clf.__class__).split('.')[-1].strip("'>")
            label = transform_label + ' -- ' + model_name
            print('EVALUATING:', label)
            print()

            # get samples and potentially remove outliers for related transforms
            train, test = get_sample(data_transformed, seed=seed, **sampling_params)
            # For oversampling
            if 'n_oversample' in sampling_params:
                train.reset_index(drop=True, inplace=True)
                # randomly add noise to training samples (for minority class)
                rows = train[train.Class==1].sample(frac=frac_add_random_noise).index
                cols = train.columns.drop(drop_cols)
                train.loc[rows, cols] += np.random.randn(len(rows), len(cols))
                
            if transform_label in ['No Outliers', 'De-Skewed - No Outliers']:
                train = cfh.remove_outliers(train, S=S, subset_rows=train.Class==0, subset_cols=train.columns.drop(drop_cols))
            print()
            plt.figure(figsize=(12,8))
            mod, x, y, stats = evaluate_classifier(train, test, clf, params, drop_cols=drop_cols)
            plt.title(label)
            results[label] = (mod,x,y,stats)
            print('===================================================')
            print('===================================================')
            print()
    return results
