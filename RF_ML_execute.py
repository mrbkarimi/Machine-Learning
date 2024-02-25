from tkinter import Tk, Label, Entry, Button
from tkinter import messagebox
from tkinter import *
import joblib
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from tkinter import ttk
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from pywaffle import Waffle
from tkinter import simpledialog
import tkinter as tk
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import matplotlib
from scipy import stats
from scipy.stats.mstats import normaltest
from scipy.stats import zscore
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.metrics import jaccard_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd
import numpy as np
import pylab as pl
import itertools
import sys
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


path = "E:/python tut/Python practices/logistic regression/Data source/"
filename = "LCRB_Single-classified.csv"
df = pd.read_csv(path + filename)


print('Developed by: Mohammad Reza Bagerzadeh Karimi')
print('\nEmail: mrbkarimi@gmail.com')


print('\n############  Please wait... #############')


x = df[['PGA/PGV', 'PGV/PGD', 'Qd/W', 'Tb/Tp', 'uy/DD']]
y = df[['Bdis_cm']]


log_Bdis = np.log(df.Bdis_cm)


sqrt_Bdis = np.sqrt(df.Bdis_cm)

y_transformed = boxcox(df.Bdis_cm)
boxcox_Bdis = y_transformed[0]
lambda_y = y_transformed[1]

# Create an empty list to store the lambda values
lambda_values = []
# Apply Box-Cox transformation to each feature
for feature2 in x:
    transformed_feature, lambda_value = boxcox(df[feature2])
    df[feature2 + '_boxcox'] = transformed_feature

    # Print lambda value for the feature
    lambda_values.append(lambda_value)  # Append the lambda value to the list


# Select the transformed features for training
x_transformed = df[[feature2 + '_boxcox' for feature2 in x.columns]]

x_train, x_test, y_train, y_test = train_test_split(
    x_transformed, y_transformed[0], test_size=0.3, random_state=10)
x_train = np.asanyarray(x_train)
y_train = np.ravel(y_train)
x_test = np.asanyarray(x_test)
y_test = np.ravel(y_test)


# Train Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=10)
rf_model.fit(x_train, y_train)

# Save the trained model using joblib
joblib.dump(rf_model, 'rf_model.pkl')

# Predictions
# x_test_s = s.transform(x_test)
y_test_pred_non_inverse = rf_model.predict(x_test)
y_train_pred_non_inverse = rf_model.predict(x_train)


# Apply inverse Box Cox transformation to predicted values
y_test_pred_inverse = inv_boxcox(y_test_pred_non_inverse, lambda_y)
y_train_pred_inverse = inv_boxcox(y_train_pred_non_inverse, lambda_y)

# Apply inverse transformation to standardized test features
y_train_inverse = inv_boxcox(y_train, lambda_y)
y_test_inverse = inv_boxcox(y_test, lambda_y)


# Create an empty array to store the inverse values
x_train_inverse = np.zeros_like(x_train)
# Inverse Box-Cox transformation for training set
for i, lambda_val in enumerate(lambda_values):
    # Apply inverse Box-Cox transformation
    x_train_inverse[:, i] = inv_boxcox(x_train[:, i], lambda_val)

# Create an empty array to store the inverse values
x_test_inverse = np.zeros_like(x_test)
# Inverse Box-Cox transformation for test set
for i, lambda_val in enumerate(lambda_values):
    # Apply inverse Box-Cox transformation
    x_test_inverse[:, i] = inv_boxcox(x_test[:, i], lambda_val)


# Evaluate the model on the training set
mse_train = metrics.mean_squared_error(y_train, y_train_pred_non_inverse)
mae_train = metrics.mean_absolute_error(y_train, y_train_pred_non_inverse)
r2_train = metrics.r2_score(y_train, y_train_pred_non_inverse)
rmse_train = np.sqrt(mse_train)
cod_train = 1 - (np.sum((y_train - y_train_pred_non_inverse) ** 2) /
                 np.sum((y_train - np.mean(y_train)) ** 2))
explained_var_train = metrics.explained_variance_score(
    y_train, y_train_pred_non_inverse)

# Evaluate the model on the training set
mse_train_inverse = metrics.mean_squared_error(
    y_train_inverse, y_train_pred_inverse)
mae_train_inverse = metrics.mean_absolute_error(
    y_train_inverse, y_train_pred_inverse)
r2_train_inverse = metrics.r2_score(y_train_inverse, y_train_pred_inverse)
rmse_train_inverse = np.sqrt(mse_train_inverse)
cod_train_inverse = 1 - (np.sum((y_train_inverse - y_train_pred_inverse) ** 2) /
                         np.sum((y_train_inverse - np.mean(y_train_inverse)) ** 2))
explained_var_train_inverse = metrics.explained_variance_score(
    y_train_inverse, y_train_pred_inverse)

# Evaluate the model on the test set
mse_test = metrics.mean_squared_error(y_test, y_test_pred_non_inverse)
mae_test = metrics.mean_absolute_error(y_test, y_test_pred_non_inverse)
r2_test = metrics.r2_score(y_test, y_test_pred_non_inverse)
rmse_test = np.sqrt(mse_test)
cod_test = 1 - (np.sum((y_test - y_test_pred_non_inverse) ** 2) /
                np.sum((y_test - np.mean(y_test)) ** 2))
explained_var_test = metrics.explained_variance_score(
    y_test, y_test_pred_non_inverse)

# Evaluate the model on the test set
mse_test_inverse = metrics.mean_squared_error(
    y_test_inverse, y_test_pred_inverse)
mae_test_inverse = metrics.mean_absolute_error(
    y_test_inverse, y_test_pred_inverse)
r2_test_inverse = metrics.r2_score(y_test_inverse, y_test_pred_inverse)
rmse_test_inverse = np.sqrt(mse_test_inverse)
cod_test_inverse = 1 - (np.sum((y_test_inverse - y_test_pred_inverse) ** 2) /
                        np.sum((y_test_inverse - np.mean(y_test_inverse)) ** 2))
explained_var_test_inverse = metrics.explained_variance_score(
    y_test_inverse, y_test_pred_inverse)

# Additional metrics
medae_test = metrics.median_absolute_error(y_test, y_test_pred_non_inverse)
# msle_test = metrics.mean_squared_log_error(y_test_bc, y_test_hat_rf)
mape_test = np.mean(np.abs((y_test - y_test_pred_non_inverse) / y_test)) * 100
huber_loss = metrics.mean_squared_error(
    y_test, y_test_pred_non_inverse, squared=False)
mbd_test = np.mean(y_test - y_test_pred_non_inverse)

###########################################################################
# Load the trained model
rf_model = joblib.load('rf_model.pkl')

# Developer information
developer_info = "Developed by Mohammad R. Bagerzadeh Karimi, Email: mrbkarimi@gmail.com"

# Function to predict bearing displacement


def predict_displacement():
    try:
        # Features of the new data
        new_pga_pgv_value = float(pga_pgv_entry.get())
        new_pgv_pgd_value = float(pgv_pgd_entry.get())
        new_f0_value = float(f0_entry.get())
        new_tb_tp_value = float(tb_tp_entry.get())
        new_uy_DD_value = float(uy_DD_entry.get())

        # Apply the appropriate transformations to the new features
        new_features = np.array([[new_pga_pgv_value, new_pgv_pgd_value,
                                  new_f0_value, new_tb_tp_value, new_uy_DD_value]])

        new_features_transformed = []
        for val in new_features:
            new_transformed_val = stats.boxcox(val, lambda_y)
            new_features_transformed.append(new_transformed_val)

        # Predict for the new features
        new_features_transformed = np.array(
            new_features_transformed).reshape(1, -1)
        predicted_target_transformed = rf_model.predict(
            new_features_transformed)

        # Inverse transform the predicted target variable
        predicted_target_inverse = inv_boxcox(
            predicted_target_transformed, lambda_y)

        # Display the predicted bearing displacement
        result_label.config(
            text=f'Predicted Bearing Displacement: {predicted_target_inverse[0]} cm')

    except ValueError:
        result_label.config(text='Please enter valid numerical values.')


# Create the main Tkinter window
window = Tk()
window.title('Bearing Displacement Predictor')

# Create entry fields for input features
Label(window, text='PGA/PGV:').grid(row=0, column=0, padx=10, pady=5)
pga_pgv_entry = Entry(window)
pga_pgv_entry.grid(row=0, column=1, padx=10, pady=5)

Label(window, text='PGV/PGD:').grid(row=1, column=0, padx=10, pady=5)
pgv_pgd_entry = Entry(window)
pgv_pgd_entry.grid(row=1, column=1, padx=10, pady=5)

Label(window, text='Q/W:').grid(row=2, column=0, padx=10, pady=5)
f0_entry = Entry(window)
f0_entry.grid(row=2, column=1, padx=10, pady=5)

Label(window, text='Tb/Tp:').grid(row=3, column=0, padx=10, pady=5)
tb_tp_entry = Entry(window)
tb_tp_entry.grid(row=3, column=1, padx=10, pady=5)

Label(window, text='uy/DD:').grid(row=4, column=0, padx=10, pady=5)
uy_DD_entry = Entry(window)
uy_DD_entry.grid(row=4, column=1, padx=10, pady=5)

# Create a button to predict displacement
predict_button = Button(
    window, text='Predict Displacement', command=predict_displacement)
predict_button.grid(row=5, column=0, columnspan=2, padx=10, pady=5)

# Create a label to display the predicted displacement
result_label = Label(window, text='')
result_label.grid(row=6, column=0, columnspan=2, padx=10, pady=5)

# Developer info label
Label(window, text=developer_info).grid(
    row=7, column=0, columnspan=2, padx=10, pady=5)

# Run the main event loop
window.mainloop()
