"""
This script implements some basic tests of the package that should be run before uploading
"""
#------------------------------------------------------------------------------
# Run interactively
#------------------------------------------------------------------------------
"""
If running program interactively, please uncomment the lines below.
But first, change the 'path_to_here' variable to your local folder!
"""
# import os
# # Manually set path of current file
# path_to_here = "/Users/muhlbach/Repositories/tbscm/src/"
# # Change path
# os.chdir(path_to_here)
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import time, random
import pandas as pd
import numpy as np

# User
from tbscm.utils import data
from tbscm.synthetic_controls import SyntheticControl as SC
from tbscm.synthetic_controls import TreeBasedSyntheticControl as TBSC
from tbscm.synthetic_controls import ElasticNetSyntheticControl as ENSC

#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------
# Simulation settings
sim_settings = {
    # General    
    "T0":500,
    "T1":50,
    "ate":1,
        
    # Y=f*
    "f":data.generate_linear_data, # generate_linear_data, generate_friedman_data_1, generate_friedman_data_2
    "beta":1,
    "include_intercept":False,
    "expand":False,
    "degree":2,
    "interaction_only":True,

    # Errors
    "eps_mean":0,
    "eps_std":1,
    "eps_cov_x":0,
    "eps_cov_y":0,
    
    # X
    "X_type":"iid",
    "X_dim":5,
    "AR_lags":3,
    }

# Start timer
t0 = time.time()

# Set seed
random.seed(1991)

#------------------------------------------------------------------------------
# Simple example
#------------------------------------------------------------------------------

# Generate data
df = data.simulate_data(**sim_settings)

# Extract data
Y = df["Y"]
W = df["W"]
X = df[[col for col in df.columns if "X" in col]]

# Instantiate SC-objects
sc = SC()
tbsc = TBSC(max_n_models=10)
ensc = ENSC(max_n_models=10)

# Fit
sc.fit(Y=Y,W=W,X=X)
print(f"Estimated ATE using SC: {np.around(sc.average_treatment_effet_,2)}")

tbsc.fit(Y=Y,W=W,X=X)
print(f"Estimated ATE using TB-SC: {np.around(tbsc.average_treatment_effet_,2)}")

ensc.fit(Y=Y,W=W,X=X)
print(f"Estimated ATE using EN-SC: {np.around(ensc.average_treatment_effet_,2)}")

# # Bootstrap
# bootstrapped_results = tbsc.bootstrap_ate()

# Stop timer
t1 = time.time()

#------------------------------------------------------------------------------
# The End
#------------------------------------------------------------------------------
print(f"""**************************************************\nCode finished in {np.round(t1-t0, 1)} seconds\n**************************************************""")


