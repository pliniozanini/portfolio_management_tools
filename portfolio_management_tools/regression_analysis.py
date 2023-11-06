# regression_analysis.py

import numpy as np
import statsmodels.api as sm

def perform_regression_analysis(dependent_variable, independent_variables):
    """
    Perform a multiple linear regression analysis
    """
    independent_variables = sm.add_constant(independent_variables)
    model = sm.OLS(dependent_variable, independent_variables).fit()
    return model.summary()

# Example usage:
# dependent_variable = ...  # Your dependent variable data (e.g., portfolio returns)
# independent_variables = ...  # Your independent variables (e.g., market returns, risk-free rate)
# result = perform_regression_analysis(dependent_variable, independent_variables)
