# monte_carlo.py

import numpy as np

def monte_carlo_simulation(returns, num_simulations, num_years, initial_investment):
    """
    Perform a Monte Carlo simulation for portfolio returns
    """
    avg_returns = returns.mean()
    std_returns = returns.std()
    
    simulation_results = np.random.normal(avg_returns, std_returns, (num_years, num_simulations))
    
    portfolio_values = np.zeros((num_years, num_simulations))
    portfolio_values[0] = initial_investment
    
    for year in range(1, num_years):
        portfolio_values[year] = portfolio_values[year - 1] * (1 + simulation_results[year])
    
    return portfolio_values

# Example usage:
# portfolio_returns = ...  # Your historical returns data
# num_simulations = 1000
# num_years = 10
# initial_investment = 100000
# simulation_results = monte_carlo_simulation(portfolio_returns, num_simulations, num_years, initial_investment)
