# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import warnings


# ignore warnings
warnings.filterwarnings('ignore')

def tune(X, y, space, scoring, 
         model, modeltype='clf', search_type='grid', n_iter_random=100,
         n_splits=5, n_repeats=3, random_state=1,
         verbose=True, display_plots=False, refit=None):
    """
    Tune hyperparameters of a machine learning model using grid search or random search.

    Parameters:
    - X (array-like): The input features.
    - y (array-like): The target variable.
    - space (dict): The hyperparameter space to search over.
    - scoring (str or callable): The scoring metric to optimize.
    - model (estimator): The machine learning model to tune.
    - modeltype (str, optional): The type of model ('clf' for classification, 'reg' for regression). Defaults to 'clf'.
    - search_type (str, optional): The type of search ('grid' for grid search, 'random' for random search). Defaults to 'grid'.
    - n_iter_random (int, optional): The number of iterations for random search. Defaults to 100.
    - n_splits (int, optional): The number of splits for cross-validation. Defaults to 5.
    - n_repeats (int, optional): The number of repeats for cross-validation. Defaults to 3.
    - random_state (int, optional): The random state for reproducibility. Defaults to 1.
    - verbose (bool, optional): Whether to display progress messages. Defaults to True.
    - display_plots (bool, optional): Whether to display plots of the search results. Defaults to False.
    - refit (str or callable, optional): The metric to use for refitting the best model. Defaults to None.

    Returns:
    - best_params (dict): The best hyperparameters found during the search.
    - best_model (estimator): The best model found during the search.
    """

    # define evaluation
    if modeltype == 'clf':
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        if type(refit) is type(None):
            refit = 'AUC'
    elif modeltype == 'reg':
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        if type(refit) is type(None):
            refit = 'neg_mean_squared_error'
  
    if verbose:
        verbosity = 1

    # define search
    if search_type == 'grid':
        search = GridSearchCV(model, space, scoring=scoring, n_jobs=-1, cv=cv, verbose=verbosity, refit=refit)
    elif search_type == 'random':
        search = RandomizedSearchCV(model, space, scoring=scoring, n_jobs=-1, cv=cv, n_iter=n_iter_random, verbose=verbosity, refit=refit)
    
    # execute search
    result = search.fit(X, y)
    # remove printing limits
    pd.set_option('display.max_columns', None)
    
    # plot results
    if display_plots:
        results_df = pd.DataFrame(result.cv_results_)
        for key, values in space.items():
            
            # group the results by the hyperparameter
            param_means = []
            param_stds = []
            for value in values:
                mask = results_df['param_' + key] == value
                param_means.append(np.mean(results_df[mask]['mean_test_score']))
                param_stds.append(np.std(results_df[mask]['mean_test_score']))
            
            # create plot with two subplots side by side
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(key)
            ax[0].plot(values, param_means)
            ax[0].set_title('Mean test scores')
            ax[0].set_xlabel(key)
            ax[0].set_ylabel('mean scores')
            padding = 0.1
            ax[0].set_ylim(max(0, min(param_means) - padding), min(1, max(param_means) + padding))

            ax[1].plot(values, param_stds)
            ax[1].set_title('Mean score std')
            ax[1].set_xlabel(key)
            ax[1].set_ylabel('score std')
            padding = 0.05
            ax[1].set_ylim(max(0, min(param_stds) - padding), min(1, max(param_stds) + padding))

            plt.show()

    # summarize result
    if verbose:
        # ANSI escape code for bold text
        bold = "\033[1m"
        # Reset ANSI escape code
        reset = "\033[0m"
        # Print the best score in bold stating the metric
        print('')
        print('Best Score (%s): %s%s%s' % (refit, bold, result.best_score_, reset))   
        
        # for each other metric, print the best score
        for metric in result.scorer_.keys():
            if metric != refit:
                print('%s: %s' % (metric, result.cv_results_['mean_test_%s' % metric][result.best_index_]))


        print('\nBest Hyperparameters:')
        for k, v in result.best_params_.items():
            print('%s: %s' % (k, v))
        print('')

    # best model
    best_model = result.best_estimator_

    return result.best_params_, best_model
