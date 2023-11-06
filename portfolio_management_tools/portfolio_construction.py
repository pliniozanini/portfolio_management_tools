# portfolio_construction.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize

def portfolio_return(weights, returns):
    """
    Calculate the expected portfolio return
    """
    return weights.T @ returns

def portfolio_volatility(weights, cov_matrix):
    """
    Calculate the expected portfolio volatility
    """
    return np.sqrt(weights.T @ cov_matrix @ weights)

def portfolio_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
    """
    Calculate the Sharpe ratio for a portfolio
    """
    portfolio_ret = portfolio_return(weights, returns)
    portfolio_vol = portfolio_volatility(weights, cov_matrix)
    return (portfolio_ret - risk_free_rate) / portfolio_vol

def minimize_portfolio_volatility(target_return, returns, cov_matrix):
    """
    Minimize portfolio volatility for a given target return
    """
    num_assets = returns.shape[1]
    initial_weights = np.repeat(1/num_assets, num_assets)
    constraints = ({'type': 'eq', 'fun': lambda weights: portfolio_return(weights, returns) - target_return},
                   {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = [(0, 1) for _ in range(num_assets)]
    result = minimize(portfolio_volatility, initial_weights, args=(cov_matrix,),
                      method='SLSQP', constraints=constraints, bounds=bounds)
    return result.x

def maximize_sharpe_ratio(returns, cov_matrix, risk_free_rate):
    """
    Maximize the Sharpe ratio for a given set of returns, covariance matrix, and risk-free rate
    """
    num_assets = returns.shape[1]
    initial_weights = np.repeat(1/num_assets, num_assets)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = [(0, 1) for _ in range(num_assets)]
    result = minimize(lambda weights: -portfolio_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate),
                      initial_weights, method='SLSQP', constraints=constraints, bounds=bounds)
    return result.x

def efficient_frontier(returns, cov_matrix, risk_free_rate, num_points=100):
    """
    Calculate the efficient frontier for a set of returns, covariance matrix, and risk-free rate
    """
    target_returns = np.linspace(returns.min(), returns.max(), num_points)
    efficient_weights = [minimize_portfolio_volatility(target_return, returns, cov_matrix) for target_return in target_returns]
    return target_returns, efficient_weights
