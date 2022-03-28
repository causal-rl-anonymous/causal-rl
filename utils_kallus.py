#from sklearn.cluster import SpectralClustering as sc
import numpy.random as rand
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge as ridge
from sklearn.linear_model import Lasso as lasso
from sklearn.linear_model import LinearRegression as ols

from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.tree import DecisionTreeRegressor as reg_tree
from sklearn.ensemble import AdaBoostRegressor as ada_reg
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.metrics import mean_squared_error as mse
import copy

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri as np2ri

robjects.numpy2ri.activate()

from rpy2.robjects.packages import importr
grf = importr('grf')
cf = grf.causal_forest

regs = [rfr(n_estimators=i) for i in [10, 20, 40, 60, 100, 150, 200]]
regs += [reg_tree(max_depth=i) for i in [5, 10, 20, 30, 40, 50]]
regs += [ada_reg(n_estimators=i) for i in [10, 20, 50, 70, 100, 150, 200]]
regs += [gbr(n_estimators=i) for i in [50, 70, 100, 150, 200]]

def get_best_for_data(X, Y, regs):
    x_train, x_test, y_train, y_test = X, X, Y, Y
    val_errs = []
    models = []
    for reg in regs:
        model = copy.deepcopy(reg)
        model.fit(x_train, y_train)
        val_errs.append(mse(y_test, model.predict(x_test)))
        models.append(copy.deepcopy(model))
    min_ind = val_errs.index(min(val_errs))
    return copy.deepcopy(models[min_ind])

linear_regs = [ols()] 
linear_regs.extend([lasso(alpha=alph) for alph in [1e-5,1e-3,1e-1,1,1e+1,1e+3,1e+5]])
linear_regs.extend([ridge(alpha=alph) for alph in [1e-5,1e-3,1e-1,1,1e+1,1e+3,1e+5]])

def run_method(X_rct, Y_rct, T_rct, X_obs, Y_obs, T_obs):
    
    #1. Estimate ^w with causal forest as family Q from observational dataset..
    _cf_model = cf(X_obs.reshape(-1,1), Y_obs.reshape(-1,1), T_obs.astype(int).reshape(-1,1), num_trees=200)
    # .. evaluated on interventional dataset ^w(Xint)
    omega_int_pred = np.array([a[0] for a in grf.predict_causal_forest(_cf_model,  X_rct.reshape(-1,1))]).ravel()

    #2.# from line above lemma 1, e^int(x) = 0.5 with interventional data and using re-weighting formula from paper
    # given q(X)Y = 2(2T -1)Y.  # Lemma 1 gives q(x)Y as unbiaised estimate of tau(Xint)
    cate_int_est = 2*np.multiply(Y_rct, 2*T_rct-1).ravel()
    assert(cate_int_est.shape == omega_int_pred.shape)
    
    #theta * x = ^tau - ^w  as solution of (1) from article. Ie linear regr with eta_est as obj
    eta_est = cate_int_est - omega_int_pred
    assert(len(eta_est.shape) == 1)
    best_eta_est_linear = get_best_for_data(X_rct.reshape(-1,1), eta_est,  regs=linear_regs)
  
    return copy.deepcopy(best_eta_est_linear), eta_est, omega_int_pred