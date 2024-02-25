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


print('Developed by: Mohammad Reza Bagerzadeh Karimi')

print('\nEmail: mrbkarimi@gmail.com')

print('\n############  Please wait... #############')

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
