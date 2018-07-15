# TODO: Add import statements
import pandas as pd
import numpy as np
from sklearn import linear_model

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('data.csv')
X = train_data.iloc[:,:6]
y = train_data.iloc[:, 6:]

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = linear_model.Lasso()

# TODO: Fit the model.
lasso_reg.fit(X, y)

# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)