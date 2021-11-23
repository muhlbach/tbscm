#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from mlregression.base.base_mlreg import BaseMLRegressor

# User
from ..utils.sanity_check import check_param_grid, check_X_Y, check_X
from ..utils.bootstrap import Bootstrap

#------------------------------------------------------------------------------
# Base Synthetic Control Group Method
#------------------------------------------------------------------------------
class BaseSyntheticControl(BaseMLRegressor):
    """
    
    """
    # -------------------------------------------------------------------------
    # Constructor function
    # -------------------------------------------------------------------------
    def __init__(self,
                 estimator,
                 param_grid=None,
                 cv_params={'scoring':None,
                            'n_jobs':None,
                            'refit':True,
                            'verbose':0,
                            'pre_dispatch':'2*n_jobs',
                            'error_score':np.nan,
                            'return_train_score':False},
                 fold_type="SingleSplit",
                 n_cv_folds=1,
                 shuffle=False,
                 test_size=0.25,
                 max_n_models=50,
                 n_cf_folds=None,
                 verbose=False,
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
            verbose=verbose)    

    # -------------------------------------------------------------------------
    # Public functions
    # -------------------------------------------------------------------------
    def fit(self,Y,W,X):
        
        # Check X and Y
        X, Y = check_X_Y(X, Y)
        
        # Find masks for the different periods pre- and post-treatment
        self.idx_T0 = W == 0
        self.idx_T1 = W == 1
        
        ## Split data into pre-treatment and post-treatment
        # Pre-treatment (Y0,X)
        self.Y_pre, self.X_pre = Y.loc[self.idx_T0], X.loc[self.idx_T0,:]
        
        # Post-treatment (Y1,X)
        self.Y_post, self.X_post = Y.loc[self.idx_T1], X.loc[self.idx_T1,:]
                
        # Estimate f in Y0 = f(X) + eps
        super().fit(X=self.X_pre,y=self.Y_pre)
        
        # Predict Y0 post-treatment
        self.Y_post_hat_ = super().predict(X=self.X_post)
        
        # Get descriptive statistics of Y both pre- and post-treatment
        self.Y_pre_mean_ = self.Y_pre.mean()
        self.Y_post_mean_ = self.Y_post.mean()
        self.Y_post_hat_mean_ = self.Y_post_hat_.mean()
        
        # Compute average treatment effect
        self.average_treatment_effet_ = self.Y_post_mean_ - self.Y_post_hat_mean_
        
        return self
        
    def calculate_average_treatment_effect(self, X_post_treatment=None):
        
        if X_post_treatment is None:
            # Recall ate from fit
            average_treatment_effect = self.average_treatment_effet_
            
        else:
            # Input validation
            X_post_treatment = check_X(X_post_treatment)
            
            # Predict Y0 post-treatment
            Y_post_hat = super().predict(X=X_post_treatment)
            
            # Average
            Y_post_hat_mean = Y_post_hat.mean()
            
            # Estimated treatment effect as the difference between means of Y1 and Y0-predicted
            average_treatment_effect = self.Y_post_mean_ - Y_post_hat_mean
                
        return average_treatment_effect
            
    def bootstrap_ate(self, bootstrap_type="circular", n_bootstrap_samples=1000, block_length=5, conf_int=0.95, X_post_treatment=None):
        
        if X_post_treatment is None:
            Y_post_hat = self.Y_post_hat_
        else:
            # Input validation
            X_post_treatment = check_X(X_post_treatment)
            
            # Predict Y0 post-treatment
            Y_post_hat = super().predict(X=X_post_treatment)
                        
        # Difference between Y1 and Y0-predicted
        Y_diff = self.Y_post - Y_post_hat
        
        # Initialize Bootstrap
        bootstrap = Bootstrap(bootstrap_type=bootstrap_type,
                              n_bootstrap_samples=n_bootstrap_samples,
                              block_length=block_length)
        
        # Generate bootstrap samples
        Y_diff_bootstrapped = bootstrap.generate_samples(x=Y_diff)
                
        # Compute mean
        Y_diff_bootstrapped_mean = Y_diff_bootstrapped.mean(axis=0)
        
        # Compute other stats from bootstrap (NOT CURRENTLY USED)
        # Y_diff_bootstrapped_std = Y_diff_bootstrapped.std(axis=0)
        # Y_diff_bootstrapped_sem = Y_diff_bootstrapped.sem(axis=0)
        
        # Confidence interval of mean
        alpha = (1-conf_int)/2
        Y_diff_bootstrapped_ci = Y_diff_bootstrapped_mean.quantile(q=[alpha,1-alpha])
        
        results_bootstrapped = {"mean" : Y_diff_bootstrapped_mean.mean(),
                                "ci_lower" : Y_diff_bootstrapped_ci.iloc[0],
                                "ci_upper" : Y_diff_bootstrapped_ci.iloc[1],
                                "mean_distribution" : Y_diff_bootstrapped_mean,
                                "difference_simulated" : Y_diff_bootstrapped,
                                }

        return results_bootstrapped
 
    
    
    
    
    
    
    
    
    
    
        