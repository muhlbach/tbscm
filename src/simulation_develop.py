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
import numpy as np
import pandas as pd
import gc
import re
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
# Plotting
from plotnine import (ggplot, aes, labs, theme_classic, theme, guide_legend, guides,
                      geom_density, geom_histogram, geom_line, geom_point, geom_hline, geom_vline,geom_smooth, geom_bar,geom_area,geom_polygon,
                      scale_y_continuous, scale_x_continuous, scale_x_datetime, scale_color_manual,scale_linetype_manual,scale_color_gradient,scale_fill_manual,
                      position_stack,after_stat,
                      element_text, element_blank,)

from matplotlib import rc
rc('text', usetex=True)

# from utils import data 

# Synthetic controls
from tbscm.utils import data
from tbscm.utils.tools import save_object_by_pickle
from tbscm.synthetic_controls import SyntheticControl as SC
from tbscm.synthetic_controls import TreeBasedSyntheticControl as TBSC
from tbscm.synthetic_controls import ElasticNetSyntheticControl as ENSC

from mlregression.mlreg import MLRegressor

#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------
# Filename
filename = "simulation_10"

# Number of simulations
n_sim = 100

# Number of covariates
p = 5

# Number of AR lags (in case)
ar_lags = 3

# Data settings
data_settings = {
    # General    
    "T0":500,
    "T1":500,
    "ate":1,
        
    # Errors
    "eps_mean":0,
    "eps_std":1,
    "eps_cov_X":0.1, # How the X's covary with each other
    "eps_cov_X_y":0.1, # How the X's covary with y
    
    # X
    "X_type":"AR", #"AR" or "cross_section"
    "X_dist":"normal", #"normal" or "uniform"
    "X_dim":p,
    "X_mean":0,
    "X_std":1,
    "X_covariance":0,
    "lower_bound":None,    
    "upper_bound":None,
    "AR_lags":ar_lags,
    "AR_coefs":1/np.exp(np.arange(1,ar_lags+1)),
    
    # Y=f*
    "f":data.generate_linear_data, # generate_linear_data, generate_friedman_data_1, generate_friedman_data_2,
    
    # For f=generate_linear_data
    "beta": 1, # 'uniform', int, float, np.ndarray, list, np.array(["uniform","uniform","uniform"]), np.array([0.25, 0.25**2, 0.25**3])
    "beta_handling":"default", # ["default", "structural", "split_order"]
    "include_intercept":False,
    "expand":True,
    "degree":2,
    "interaction_only":False,
    "enforce_limits":False,
    
    # For f=friedman no specific arguments
    
    }

# Estimation settings
est_settings = {
    'cv_params':{'scoring':None,
           'n_jobs':None,
           'refit':True,
           'verbose':0,
           'pre_dispatch':'2*n_jobs',
           'error_score':np.nan,
           'return_train_score':False},
    'fold_type':"SingleSplit",
    'n_cv_folds':1,
    'shuffle':False,
    'test_size':0.25,
    'max_n_models':10,
    'n_cf_folds':None,
    'verbose':False,
    }

#Max abs radius of distribution plot
max_x_lim = 1

#------------------------------------------------------------------------------
# Derived
#------------------------------------------------------------------------------
sim_num = re.sub('\D', '', filename)

# True average treatment effect
true_ate = data_settings["ate"]

#------------------------------------------------------------------------------
# Defaults
#------------------------------------------------------------------------------
# Start timer
t0 = time.time()

# Set seed
random.seed(1991)

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
# Pre-allocate
df_results = pd.DataFrame()

for sim in range(1,n_sim+1):
    if sim % (n_sim/10) == 0:
        print(f"Beginning simulation {sim}/{n_sim}")

    # # Generate data
    df = data.simulate_data(**data_settings)

    # # Extract data
    Y = df["Y"]
    Ystar = df["Ystar"]
    W = df["W"]
    Y0 = Y - W*true_ate # Same as Ystar+df["U"]
    X = df[[col for col in df.columns if "X" in col]]
    
    X.describe()
    X.corr()
    
    # Get indices
    idx_pre = W==0
    idx_post = W==1

    # Get the truth in the control state
    Y0star_post = Ystar[idx_post]

    # Orable
    yhat_oracle = Y0star_post.copy()
    oracle_ate = (Y.loc[idx_post]-yhat_oracle).mean()
    
    # Instantiate SC-objects
    sc = SC()
    ensc = ENSC(**est_settings)
    tbsc = TBSC(**est_settings)
        
    # Fit objects
    sc.fit(Y=Y,W=W,X=X)
    ensc.fit(Y=Y,W=W,X=X)
    tbsc.fit(Y=Y,W=W,X=X)

    # Compute counterfactual mse that would have happened had the treatment effect been zero
    sc_mse = mean_squared_error(y_true=Y0star_post, y_pred=sc.Y_post_hat_)
    ensc_mse = mean_squared_error(y_true=Y0star_post, y_pred=ensc.Y_post_hat_)
    tbsc_mse = mean_squared_error(y_true=Y0star_post, y_pred=tbsc.Y_post_hat_)
    
    # Store results for this particular simulation
    df_results_temp = pd.DataFrame({"ate_true" : data_settings["ate"],
                                    "ate_oracle" : oracle_ate,
                                    "ate_sc" : sc.average_treatment_effet_,
                                    "ate_ensc" : ensc.average_treatment_effet_,
                                    "ate_tbsc" : tbsc.average_treatment_effet_,                                    
                                    "mse_sc" : sc_mse,
                                    "mse_ensc" : ensc_mse,
                                    "mse_tbsc" : tbsc_mse,
                                    },
                                    index=[sim-1])

    # Append results
    df_results = df_results.append(df_results_temp)

    # House-keeping
    del df, Y, W, X, df_results_temp, sc, tbsc, ensc
    if sim % 10 == 0:
        gc.collect()

#------------------------------------------------------------------------------
# Cross-validation error
#------------------------------------------------------------------------------
print("\nMSE:\n", df_results[[col for col in df_results.columns if ("mse" in col) and ("true" not in col)]].mean())

#------------------------------------------------------------------------------
# Pre-liminiary analysis of estimated ate
#------------------------------------------------------------------------------
df_results_pivot = df_results.melt(value_vars=[col for col in df_results.columns if ("ate" in col) and ("true" not in col)],
                          var_name="method",
                          value_name="ate")

# Add bias
df_results_pivot["bias"] = df_results_pivot["ate"] - true_ate

df_results_overview = df_results_pivot.groupby(by=["method"]).agg(
    mse=("ate", lambda x: ((x-true_ate)**2).mean()),
    bias2=("ate", lambda x: (x.mean()-true_ate)**2),
    var=("ate", lambda x: ((x-x.mean())**2).mean()) #x.var() vs. ((x-x.mean())**2).mean()
    )

print("\nBias-variance trade-off:\n", df_results_overview)

# df_results_overview["mse"] - df_results_overview["bias2"]+df_results_overview["var"]


#------------------------------------------------------------------------------
# Plot results
#------------------------------------------------------------------------------
# Melt
print("\nCentered ATE (bias):", df_results_pivot.groupby("method")["bias"].mean(), "\n")

method_mapper = {"ate_sc" : "SC",
                 "ate_ensc" : "Elastic net SC",
                 "ate_tbsc" : "Tree-based SC",
                 }

df_plot = df_results_pivot.copy()
df_plot['method'] = pd.Categorical(values=df_plot['method'].map(method_mapper),
                                   categories=list(method_mapper.values()),
                                   ordered=True)

win_qnt = 0.0
symmetric_lim = min(max([abs(df_results_pivot['bias'].quantile(win_qnt)),
                     abs(df_results_pivot['bias'].quantile(1-win_qnt))]),
                    max_x_lim)

x_lim = [-symmetric_lim, +symmetric_lim]

# Plot
gg = (
    # Initialize plot
    ggplot(data=df_plot, mapping=aes(x="bias", y=after_stat('density'), fill="method"))

    # Show histogram with density as density
    + geom_density(alpha=0.4, size=1)
    
    + geom_vline(aes(xintercept=0), linetype="dashed", size=1.25)
    
    # Labs
    + labs(
        title=f"Distribution of estimated ATE in simulation {sim_num}",
        x='$\\hat{\\tau}-\\tau_{0}$',
        y="",
        size=14
        )
    
    # Axis
    + scale_y_continuous(expand=(0,0), )
    + scale_x_continuous(expand=(0,0), limits=x_lim)
    
    # Theme
    + theme_classic()
    
    + theme(
        # Title
        title=element_text(size=24),
    
        # Legend
        legend_position="bottom",
        legend_title=element_blank(),
        legend_box_spacing=0.85,
        legend_text=element_text(size=14),
    
        # Axis
        axis_title=element_text(size=20),
        axis_text_x=element_text(size=14, angle=0),
        axis_text_y=element_text(size=14),
    
        # General
        text=element_text(family="Times New Roman",
                          style="normal",
                          weight="normal"),
        )
    )
gg
#------------------------------------------------------------------------------
# Save to "Settings master"
#------------------------------------------------------------------------------
# Construct df with settings
df_settings = pd.DataFrame.from_dict({**data_settings,
                                      **est_settings,
                                      **{"n_sim":n_sim}}, orient='index', columns=[filename])

try:
    # Read all 
    df_settings_all = pd.read_csv(filepath_or_buffer="../settings/"+"settings_master"+".csv", index_col=0)
       
    # Merge
    df_settings_all = df_settings_all.merge(df_settings, left_index=True, right_index=True)

except:
    # Copy
    df_settings_all = df_settings

#------------------------------------------------------------------------------
# Save to "Results master"
#------------------------------------------------------------------------------
# Construct df with settings
df_results_overview_pivot = df_results_overview.melt(var_name="stat",
                                                     value_name=filename,
                                                     ignore_index=False).reset_index()

try:
    # Read all 
    df_results_all = pd.read_csv(filepath_or_buffer="../results/"+"results_master"+".csv")
       
    # Merge
    df_results_all = df_results_all.merge(df_results_overview_pivot, on=["method", "stat"])

except:
    # Copy
    df_results_all = df_results_overview_pivot

#------------------------------------------------------------------------------
# The End
#------------------------------------------------------------------------------
# Stop timer
t1 = time.time()


print(f"""**************************************************\nCode finished in {np.round(t1-t0, 1)} seconds\n**************************************************""")
