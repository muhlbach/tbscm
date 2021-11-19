# *** ATTENTION ***
Don't immidiately run `pip install tbscm`. See Section _Installation_.

# Tree-based Synthetic Control Methods (tbscm)

This package implements the Tree-based Synthetic Control Methods (tbscm) from Mühlbach & Nielsen (2021), see https://arxiv.org/abs/1909.03968. 

The method is essentially a nonparametric extension of the classic synthetical control group estimator proposed by Alberto Abadie and co-authors.

Please contact the authors below if you find any bugs or have any suggestions for improvement. Thank you!

Author: Nicolaj Søndergaard Mühlbach (n.muhlbach at gmail dot com, muhlbach at mit dot edu) 

## Code dependencies
This code has the following dependencies:

- Python >=3.6
- numpy >=1.19
- pandas >=1.3
- mlregression >=0.1.6

Note that `mlregression` in turn depends on
- scikit-learn >=1
- scikit-learn-intelex >= 2021.3
- daal >= 2021.3
- daal4py >= 2021.3
- tbb >= 2021.4
- xgboost >=1.3
- lightgbm >=3.2

## Installation
Before calling `pip install tbscm`, we recommend installing `mlregression`. For installation of `mlregression` and dependensies, please visit: https://pypi.org/project/mlregression/.

## Usage
We demonstrate the use of __mlregression__ below, using random forests, xgboost, and lightGBM as underlying regressors.

```python
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import time, random
import numpy as np

# User
from tbscm.utils import data
from tbscm.synthetic_controls import SyntheticControl as SC
from tbscm.synthetic_controls import TreeBasedSyntheticControl as TBSC
from tbscm.synthetic_controls import ElasticNetSyntheticControl as ENSC

#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------
# Number of covariates
p = 2
ar_lags = 3

# Number of max models to run
max_n_models = 5

# Data settings
data_settings = {
    # General    
    "T0":500,
    "T1":500,
    "ate":1,
        
    # Errors
    "eps_mean":0,
    "eps_std":1,
    "eps_cov_xx":0, # How the X's covary with each other
    "eps_cov_yy":0.1, # How the X's covary with y
    
    # X
    "X_type":"AR",
    "X_dist":"normal",
    "X_dim":p,
    "mu":0,
    "sigma":1,
    "covariance":0,
    "AR_lags":ar_lags,
    "AR_coefs":1/np.exp(np.arange(1,ar_lags+1)),
    
    # Y=f*
    "f":data.generate_linear_data, # generate_linear_data, generate_friedman_data_1, generate_friedman_data_2,
    }

# Start timer
t0 = time.time()

# Set seed
random.seed(1991)

#------------------------------------------------------------------------------
# Simple example
#------------------------------------------------------------------------------
# Generate data
df = data.simulate_data(**data_settings)

# True ate
ate = data_settings["ate"]

# Extract data
Y = df["Y"]
W = df["W"]
X = df[[col for col in df.columns if "X" in col]]

# Instantiate SC-objects
sc = SC()
tbsc = TBSC(max_n_models=max_n_models)
ensc = ENSC(max_n_models=max_n_models)

# Fit
sc.fit(Y=Y,W=W,X=X)
print(f"Estimated ATE using SC: {np.around(sc.average_treatment_effet_,2)}")

tbsc.fit(Y=Y,W=W,X=X)
print(f"Estimated ATE using TB-SC: {np.around(tbsc.average_treatment_effet_,2)}")

ensc.fit(Y=Y,W=W,X=X)
print(f"Estimated ATE using EN-SC: {np.around(ensc.average_treatment_effet_,2)}")

# Bootstrap
bootstrapped_results = tbsc.bootstrap_ate()
```