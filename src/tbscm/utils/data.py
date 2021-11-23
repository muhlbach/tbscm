#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tools.tools import add_constant
from scipy.stats import norm

# User
from .exceptions import WrongInputException

###############################################################################
# Main
###############################################################################

#------------------------------------------------------------------------------
# Tools
#------------------------------------------------------------------------------
def get_colnames(x,prefix="X"):
    try:
        dim = x.shape[1]
        colnames = [prefix+str(j) for j in np.arange(start=1,stop=dim+1)]
    except IndexError:
        colnames = [prefix]
        
    return colnames

def convert_to_dict_series(Yobs=None,Ytrue=None,Y0=None,Y1=None,W=None):
    # Save local arguments
    args = locals()
    
    # Convert values to series with appropriate names
    args = {k: pd.Series(v, name=k) for k,v in args.items() if v is not None}
    
    return args

def convert_to_dict_df(X=None):
    # Save local arguments
    args = locals()
    
    # Convert values to series with appropriate names
    args = {k: pd.DataFrame(v, columns=get_colnames(x=v,prefix=k)) for k,v in args.items() if v is not None}
    
    return args

def convert_normal_to_uniform(x, mu="infer", sigma="infer", lower_bound=0, upper_bound=1, n_digits_round=2):
    """ See link: https://math.stackexchange.com/questions/2343952/how-to-transform-gaussiannormal-distribution-to-uniform-distribution
    """
    # Convert to np and break link
    x = np.array(x.copy())   
       
    if mu=="infer":
        mu = np.mean(x, axis=0).round(n_digits_round)
    if sigma=="infer":
        sigma = np.sqrt(np.var(x, axis=0)).round(n_digits_round)
    
    # Get CDF
    x_cdf = norm.cdf(x=x, loc=mu, scale=sigma)
    
    # Transform
    x_uni = (upper_bound-lower_bound)*x_cdf - lower_bound
    
    return x_uni
    


#------------------------------------------------------------------------------
# Generate X data
#------------------------------------------------------------------------------
def multivariate_normal(N,p,mu,sigma,covariance,lower_limit=None, upper_limit=None):
    
    if (lower_limit is None) and (upper_limit is None):
        
        # Covariance matrix
        cov_diag = np.diag(np.repeat(a=sigma**2, repeats=p))
        cov_off_diag = np.ones(shape=(p,p)) * covariance
        np.fill_diagonal(a=cov_off_diag, val=0)
        cov_mat = cov_diag + cov_off_diag

        X = pd.DataFrame(np.random.multivariate_normal(mean=np.repeat(a=mu, repeats=p), 
                                                        cov=cov_mat,
                                                        size=N))

    else:
    
        valid_N = 0
        X = pd.DataFrame()
        
        while valid_N<N:
            
            # Generate temporary data without limits
            X_temp = multivariate_normal(N=N,p=p,mu=mu,sigma=sigma,covariance=covariance,lower_limit=None,upper_limit=None)
            
            if lower_limit is None:
                lower_limit = -np.inf
            elif upper_limit is None:
                upper_limit = +np.inf

            if lower_limit>=upper_limit:
                raise Exception(f"Lower limit (= {lower_limit}) cannot exceed upper limit (= {upper_limit})")
            
            # Invalid indices
            invalid_idx = (X_temp < lower_limit).any(axis=1) | (X_temp > upper_limit).any(axis=1)
                        
            # Remove rows that exceed limits
            X_temp = X_temp.loc[~invalid_idx,:]
            
            # Append
            X = X.append(X_temp, ignore_index=True)
    
            valid_N = len(X)
            
        X = X.iloc[0:N,:]
            
    return X


def generate_ar_process(T,
                        p,
                        AR_lags,
                        AR_coefs,
                        burnin=50,
                        intercept=0,
                        mu0=0,
                        sigma0=1,
                        coef_on_error=1,
                        **kwargs):
    
    # Fix AR coefs; flip order and reshape to comformable shape
    AR_coefs = np.flip(AR_coefs).reshape(-1,1)

    # Generate errors
    errors = kwargs.get('errors', np.random.multivariate_normal(mean=np.ones(p), 
                                                                cov=np.identity(p),
                                                                size=T))    

    # Generate errors for burn-in period
    errors_burnin = np.random.multivariate_normal(mean=np.mean(errors,axis=0), 
                                                  cov=np.cov(errors.T),
                                                  size=burnin)

    errors_all = np.concatenate((errors_burnin,errors))

    # Generate initial value(s)
    X = mu0 + sigma0 * np.random.randn(AR_lags,p)

    # Simulate AR(p) with burn-in included
    for b in range(burnin+T):
        X = np.concatenate((X,
                            intercept + AR_coefs.T @ X[0:AR_lags,:] + coef_on_error * errors_all[b,0:p]),
                           axis=0)

    # Return only the last T observations (we have removed the dependency on the initial draws)
    return X[-T:,]

def generate_cross_sectional_data(N,
                                  p,
                                  distribution="normal",
                                  mu=None,
                                  sigma=None,
                                  covariance=None,
                                  lower_bound=None,
                                  upper_bound=None,
                                  dtype="np.darray",
                                  **kwargs):
        
    DISTRIBUTION_ALLOWED = ["normal", "uniform"]

    # Generate X
    if distribution=="normal":
        if (mu is None) or (sigma is None) or (covariance is None):
            raise Exception("When 'distribution'=='normal', both 'mu', 'sigma', and 'covariance' must be provided and neither can be None")
        
        X = multivariate_normal(N=N,
                                p=p,
                                mu=mu,
                                sigma=sigma,
                                covariance=covariance,
                                lower_limit=lower_bound,
                                upper_limit=upper_bound)
        
    elif distribution=="uniform":
        if (lower_bound is None) or (upper_bound is None):
            raise Exception("When 'distribution'=='uniform', both 'lower_bound' and 'upper_bound' must be provided and neither can be None")
        
        # Draw from uniform distribution
        X = np.random.uniform(low=lower_bound,
                              high=upper_bound,
                              size=(N,p))
    else:
        raise WrongInputException(input_name="distribution",
                                  provided_input=distribution,
                                  allowed_inputs=DISTRIBUTION_ALLOWED)
                   
    if dtype=="np.darray":
        X = np.array(X)
    elif dtype=="pd.DataFrame":
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
    return X

def generate_errors(N=1000, p=5, mu=0, sigma=1, cov_X=0.25, cov_X_y=0.5):

    # Number of dimensions including y
    n_dim = p+1

    ## Construct variance-covariance matrix
    # Construct diagonal with variance = sigma^2
    cov_diag = np.diag(np.repeat(a=sigma**2, repeats=n_dim))
    
    ## Construct off-diagonal with covariances
    # Fill out for X (and y)
    cov_off_diag = np.ones(shape=(n_dim,n_dim)) * cov_X
    
    # Update y entries
    cov_off_diag[p,:] = cov_off_diag[:,p] = cov_X_y
    
    # Set diagonal to zero
    np.fill_diagonal(a=cov_off_diag, val=0)
    
    # Update final variance-covariance matrix
    cov_mat = cov_diag + cov_off_diag

    # Generate epsilon
    eps = np.random.multivariate_normal(mean=np.repeat(a=mu, repeats=n_dim), 
                                        cov=cov_mat,
                                        size=N)    

    return eps

#------------------------------------------------------------------------------
# Generate f_star = E[Y|X=x]
#------------------------------------------------------------------------------
def _solve_meta_problem(A,B,w):
    """
    Solve diag(X @ A') = B @ w for X such that X_ij>=0 and sum_j(X_ij)==1 for all i
    """
    # Vectorize weights
    w = _vectorize_beta(beta=w,x=B)
    
    # Set up variable to solve for
    X = cp.Variable(shape=(A.shape))
        
    # Set up constraints
    constraints = [X >= 0,
                   X @ np.ones(shape=(A.shape[1],)) == 1
                   ]
    
    # Set up objective function
    objective = cp.Minimize(cp.sum_squares(cp.diag(X @ A.T) - B @ w))
    
    # Instantiate
    problem = cp.Problem(objective=objective, constraints=constraints)
    
    # Solve (No need to specify solver because by default CVXPY calls the solver most specialized to the problem type)
    problem.solve(verbose=False)
    
    return X.value
    
def _vectorize_beta(beta,x):
    """
    Turn supplied beta into an appropriate shape
    """
    if isinstance(beta, (int, float, np.integer)):
        beta = np.repeat(a=beta, repeats=x.shape[1])        
    elif isinstance(beta, np.ndarray):
        if len(beta)<x.shape[1]:
            beta = np.tile(A=beta, reps=int(np.ceil(x.shape[1]/len(beta))))
        # Shorten potentially
        beta = beta[:x.shape[1]]
    elif isinstance(beta, str):
        if beta=="uniform":
            beta = np.repeat(a=1/x.shape[1], repeats=x.shape[1])            
    else:
        raise WrongInputException(input_name="beta",
                                  provided_input=beta,
                                  allowed_inputs=[int, float, str, np.ndarray, np.integer])       
        
    # Make sure beta has the right dimensions
    beta = beta.reshape(-1,)        
    
    if x.shape[1]!=beta.shape[0]:
            raise Exception(f"Beta is {beta.shape}-dim vector, but X is {x.shape}-dim matrix")
    
    return beta


def generate_linear_data(x,
                         beta=1,
                         beta_handling="default",
                         include_intercept=False,
                         expand=False,
                         degree=2,
                         interaction_only=False,
                         enforce_limits=False,
                         tol_fstar=100,
                         **kwargs):
    """
    Parameters
    ----------
    x : np.array or pd.DataFrame
        Exogeneous data
    beta : int, list-type or array, optional
        Coefficients to be multiplied to x. The default is 1.
    beta_handling : str, optional
        How to handle beta. The default is "default".
        if "default", use x'beta
        if "structural", make it look like some beta was multiplied to x, where it fact we use clever weights
        
        
    include_intercept : bool, optional
        Add intercept/bias term to x. The default is False.
    expand : bool, optional
        Add higher-order terms of x. The default is False.
    degree : int, optional
        Degree of higher-order terms if expand==True. The default is 2.
    interaction_only : bool, optional
        Whether to focus on interactions when expand==True or also higher order polynomials. The default is False.
    enforce_limits : bool, optional
        Enforce f_star to be min(x) <= max(x). The default is False.
    tol_fstar : float, optional
        Tolerance when beta_handling="structural". The default is 100.

    Returns
    -------
    f_star : np.array
        Conditional mean of Y
    """

    #
    BETA_HANDLING_ALLOWED = ["default", "structural", "split_order"]
    
    # Convert to np and break link
    x = np.array(x.copy())    

    # Convert extrama points of X
    if enforce_limits:
        x_min, x_max  = np.min(x, axis=1), np.max(x, axis=1)

    # Series expansion of X
    if expand:        
        if degree<2:
            raise Exception(f"When polynomial features are generated (expand=True), 'degree' must be >=2. It is curently {degree}")
        
        # Instantiate 
        polynomialfeatures = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False, order='C')
    
        # Expand x
        x_poly = polynomialfeatures.fit_transform(x)[:,x.shape[1]:]
        
        # Concatenate
        x_all = np.concatenate((x,x_poly), axis=1)
        
    else:
        x_all = x
        
    # Include a constant in X
    if include_intercept:
        x = add_constant(data=x, prepend=True, has_constant='skip')
        
    # Different ways to generating beta and fstar
    if beta_handling=="default":
        # Make beta a conformable vector
        beta = _vectorize_beta(beta=beta,x=x_all)
                
        # Generate fstar=E[y|X=x]
        f_star = x_all @ beta

    elif beta_handling=="structural":
        """
        Constrcut Y=f_star, such that
        f_star = diag(WX')=X_all*beta_uniform, with with summing to one per j and all non-negative.
        """
        # Get tricky weight matrix, solving diag(WX')=X_all*beta_uniform
        weights = _solve_meta_problem(A=x, B=x_all, w="uniform")        

        # Generate fstar=E[y|X=x]
        f_star = np.diagonal(weights @ x.T)
        
        # Fact check this
        f_star_check = x_all @ _vectorize_beta(beta="uniform",x=x_all)

        if np.sum(f_star-f_star_check) > tol_fstar:
            raise Exception("Trickiness didn't work as differences are above tolerance")        
        
    elif beta_handling=="split_order":    
        """
        Apply different beta to each higher-order term, forinstance X*b1 + X^2*b2 + X^3*b3, where beta=[b1,b2,b3]
        """
        if isinstance(beta, (int, float, str, np.integer)):
            raise Exception("Whenever 'beta_handling'='split_order', then 'beta' cannot be either (int, float, str)")
        elif len(beta)!=degree:
            raise Exception(f"beta is of length {len(beta)}, but MUST be of length {degree}")
        if not expand:
            raise Exception("Whenever 'beta_handling'='split_order', then 'expand' must be True")
        
        # First-order beta
        beta_first_order = _vectorize_beta(beta=beta[0],x=x)
        
        # Higher-order beta
        beta_higher_order = np.empty(shape=(0,))

        # Initialize
        higher_order_col = 0
        for higher_order in range(2,degree+1):

            # Instantiate 
            poly_temp = PolynomialFeatures(degree=higher_order, interaction_only=interaction_only, include_bias=False, order='C')
    
            # Expand x
            x_poly_temp = poly_temp.fit_transform(x)[:,x.shape[1]+higher_order_col:]

            # Generate temporary betas for this degree of the expansion
            beta_higher_order_temp = _vectorize_beta(beta=beta[higher_order-1],x=x_poly_temp)
                
            # Append betas
            beta_higher_order = np.append(arr=beta_higher_order, values=beta_higher_order_temp)
        
            # Add column counter that governs which columns to match in X
            higher_order_col += x_poly_temp.shape[1]
        
        # Generate fstar=E[y|X=x]
        f_star = x @ beta_first_order + x_poly @ beta_higher_order
        
    else:
        raise WrongInputException(input_name="beta_handling",
                                  provided_input=beta_handling,
                                  allowed_inputs=BETA_HANDLING_ALLOWED)  
        
    # Reshape for conformity
    f_star = f_star.reshape(-1,)
    
    if enforce_limits:
        f_star = np.where(f_star<x_min, x_min, f_star)
        f_star = np.where(f_star>x_max, x_max, f_star)
    
    return f_star
    

def generate_friedman_data_1(x, **kwargs):
    
    # Convert to np and break link
    x = np.array(x.copy())    

    # Sanity check
    if x.shape[1]<5:
        raise Exception(f"Friedman 1 requires at least 5 regresors, but only {x.shape[1]} are provided in x")

    # Generate fstar=E[y|X=x]
    f_star = 0.1*np.exp(4*x[:,0]) + 4/(1+np.exp(-20*(x[:,1]-0.5))) + 3*x[:,2] + 2*x[:,3] + 1*x[:,4]
    
    # Reshape for conformity
    f_star = f_star.reshape(-1,)
    
    return f_star

def generate_friedman_data_2(x, **kwargs):
    
    # Convert to np and break link
    x = np.array(x.copy())    

    # Sanity check
    if x.shape[1]<5:
        raise Exception(f"Friedman 2 requires at least 5 regresors, but only {x.shape[1]} are provided in x")

    # Generate fstar=E[y|X=x]
    f_star = 10*np.sin(np.pi*x[:,0]*x[:,1]) + 20*(x[:,2]-0.5)**2 + 10*x[:,3] + 5*x[:,4]
    
    # Reshape for conformity
    f_star = f_star.reshape(-1,)
    
    return f_star

#------------------------------------------------------------------------------
# Simulate data
#------------------------------------------------------------------------------
def simulate_data(f,
                  T0=500,
                  T1=50,
                  X_type="cross_section",
                  X_dist="normal",
                  X_dim=5,
                  X_mean=0,
                  X_std=1,
                  X_covariance=0,
                  ate=1,
                  eps_mean=0,
                  eps_std=1,
                  eps_cov_X=0,
                  eps_cov_X_y=0,
                  **kwargs):

    # Total number of time periods
    T = T0 + T1

    ## Step 1: Generate errors (because they might be used in the generation of X)

    # Generate errors
    errors = generate_errors(N=T, p=X_dim, mu=eps_mean, sigma=eps_std, cov_X=eps_cov_X, cov_X_y=eps_cov_X_y)
    
    # Generate covariates
    if X_type=="AR":
        X = generate_ar_process(
            T=T,
            p=X_dim,
            errors=errors,
            **kwargs
            )
                
    elif X_type=="cross_section":
        X = generate_cross_sectional_data(N=T,
                                          p=X_dim,
                                          mu=X_mean,
                                          sigma=X_std,
                                          covariance=X_covariance,
                                          distribution=X_dist,
                                          **kwargs
                                          )
                
    # Generate W
    W = np.repeat((0,1), (T0,T1))

    # Generate Ystar
    Ystar = f(x=X, **kwargs)

    # Generate Y
    Y = Ystar + ate*W + errors[:,-1]

    # Collect data
    df = pd.concat(objs=[pd.Series(data=Y,name="Y"),
                         pd.Series(data=W,name="W"),
                         pd.DataFrame(data=X,columns=[f"X{d}" for d in range(X.shape[1])]),
                         pd.Series(data=Ystar,name="Ystar"), 
                         pd.Series(data=errors[:,-1],name="U"), 
                         ],
                    axis=1)
        
    return df                                  