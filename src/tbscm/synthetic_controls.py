#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import ElasticNet

# User
from .base.base_synthetic_controls import BaseSyntheticControl
from .estimator.constrained_ols import ConstrainedOLS

#------------------------------------------------------------------------------
# Local parameters
#------------------------------------------------------------------------------
cv_params={'scoring':None,
       'n_jobs':None,
       'refit':True,
       'verbose':0,
       'pre_dispatch':'2*n_jobs',
       'error_score':np.nan,
       'return_train_score':False}
fold_type="SingleSplit"
n_cv_folds=1
shuffle=False
test_size=0.25
max_n_models=50
n_cf_folds=None
verbose=False

#------------------------------------------------------------------------------
# Ordinary Synthetic Control Group Method
#------------------------------------------------------------------------------
class SyntheticControl(BaseSyntheticControl):
    """
    This class estimates the average treatment effects by constructing a synthetic control group using constrained OLS
    """
    # --------------------
    # Constructor function
    # --------------------
    # HERE!
    def __init__(self,
                 estimator=ConstrainedOLS(),
                 param_grid={'coefs_lower_bound':0,
                             'coefs_lower_bound_constraint':">=",
                             'coefs_sum_bound':1,
                             'coefs_sum_bound_constraint':"<=",},
                 cv_params=cv_params,
                 fold_type=fold_type,
                 n_cv_folds=n_cv_folds,
                 shuffle=shuffle,
                 test_size=test_size,
                 max_n_models=max_n_models,
                 n_cf_folds=n_cf_folds,
                 verbose=verbose,
                 ):
        super().__init__(
            estimator=estimator,
            param_grid=param_grid,
            cv_params=cv_params,
            fold_type=fold_type,
            n_cv_folds=n_cv_folds,
            shuffle=shuffle,
            test_size=test_size,
            max_n_models=max_n_models,
            n_cf_folds=n_cf_folds,
            verbose=verbose,
            )



#------------------------------------------------------------------------------
# Tree-Based Synthetic Control Group Method
#------------------------------------------------------------------------------
class TreeBasedSyntheticControl(BaseSyntheticControl):
    """
    This class estimates the average treatment effects by constructing a synthetic control group using Random Forests
    """
    # --------------------
    # Constructor function
    # --------------------
    def __init__(self,
                 estimator="RandomForestRegressor",
                 param_grid=None,
                cv_params=cv_params,
                fold_type=fold_type,
                n_cv_folds=n_cv_folds,
                shuffle=shuffle,
                test_size=test_size,
                max_n_models=max_n_models,
                n_cf_folds=n_cf_folds,
                verbose=verbose,
                ):
        super().__init__(
            estimator=estimator,
            param_grid=param_grid,
            cv_params=cv_params,
            fold_type=fold_type,
            n_cv_folds=n_cv_folds,
            shuffle=shuffle,
            test_size=test_size,
            max_n_models=max_n_models,
            n_cf_folds=n_cf_folds,
            verbose=verbose,
            )

            
#------------------------------------------------------------------------------
# Elastic Net Synthetic Control Group Method
#------------------------------------------------------------------------------
class ElasticNetSyntheticControl(BaseSyntheticControl):
    """
    This class estimates the average treatment effects by constructing a synthetic control group using Elastic Net
    """
    # --------------------
    # Constructor function
    # --------------------
    def __init__(self,
                 estimator="ElasticNet",
                 param_grid=None,
                 cv_params=cv_params,
                 fold_type=fold_type,
                 n_cv_folds=n_cv_folds,
                 shuffle=shuffle,
                 test_size=test_size,
                 max_n_models=max_n_models,
                 n_cf_folds=n_cf_folds,
                 verbose=verbose,
                 ):
        super().__init__(
            estimator=estimator,
            param_grid=param_grid,
            cv_params=cv_params,
            fold_type=fold_type,
            n_cv_folds=n_cv_folds,
            shuffle=shuffle,
            test_size=test_size,
            max_n_models=max_n_models,
            n_cf_folds=n_cf_folds,
            verbose=verbose,
            )
    
    
        