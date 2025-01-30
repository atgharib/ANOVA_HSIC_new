from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target
import copy 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


import os
import pickle
import time
from openpyxl import Workbook
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest, f_classif, f_regression, RFECV
from sklearn.linear_model import Lasso
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.svm import SVC, SVR
from pathlib import Path
import gc
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from real_datasets import load_dataset

from HSICNet.HSICNet import *
from HSICNet.HSICFeatureNet import *
from explainers.L2x_reg import *
from invase import INVASE
import sys 

def train_svm(X_train, y_train, X_test, y_test):
    """
    Train an SVM or SVR model with imputation for missing values.

    Parameters:
        X_train: Training feature matrix
        y_train: Training labels
        X_test: Testing feature matrix
        y_test: Testing labels

    Returns:
        best_model: Trained model after hyperparameter tuning
        best_params: Best hyperparameters from GridSearchCV
        score: Performance score (accuracy for classification, RMSE for regression)
    """

    # Check the type of the target variable
    target_type = type_of_target(y_train)
    is_classification = target_type in ["binary", "multiclass"]

    # Define the parameter grid
    param_grid = {
        "svm__C": [0.1, 1, 10],  # Regularization parameter
    }

    # Choose the model
    model = SVC() if is_classification else SVR()

    # Create a pipeline with an imputer and the SVM/SVR model
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),  # Impute missing values
        ("svm", model)
    ])

    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy" if is_classification else "neg_mean_squared_error")
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    if is_classification:
        score = accuracy_score(y_test, y_pred)
    else:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        score = rmse

    print("Best Parameters:", best_params)
    print("Performance Score:", score)

    return best_model, best_params, score

# Ensure a directory exists for saving models
os.makedirs("trained_models", exist_ok=True)

# Define the list of feature selectors
feature_selectors = ["L2X", "INVASE"]

# Initialize an Excel workbook to store global importance values
wb = Workbook()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def memory_cleaning():
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()

epoch=500
layers = [200, 300, 400, 300, 200]
feature_layers = [20, 50, 100, 50, 20]
act_fun_featlayer = torch.nn.SELU
act_fun_layer = torch.nn.Sigmoid

if __name__ == '__main__':

    # dataset_names = ["breast_cancer", "sonar", "nomao", "breast_cancer_wisconsin", "skillcraft", "ionosphere", \
                    #  "sml", "pol",'parkinson', 'keggdirected', "pumadyn32nm", "crime", "gas",'autos', 'bike', 'keggundirected']
    dataset_names = ["sonar"]
    # Main running part of the script
    for dataset_name in dataset_names:
        print(f"\nProcessing dataset: {dataset_name}")
        try:
            X, y = load_dataset(dataset_name)
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            continue

        # Determine if the dataset is for classification or regression
        mode = "classification" if type_of_target(y) in ["binary", "multiclass"] else "regression"
        if mode == "classification": 
            y = LabelEncoder().fit_transform(y) 
        #else: continue

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        n, d = X_train.shape


        model = RandomForestRegressor(n_estimators=500)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        exp_func = model.predict


        # Convert the data to PyTorch tensors
        X_tensor_train = torch.tensor(X_train, dtype=torch.float32).to(device=device)
        y_tensor_train = torch.tensor(y_train, dtype=torch.float32).to(device=device) if mode != "classification" else \
                    torch.tensor(y_train, dtype=torch.int32).to(device=device)
        
        X_tensor_test = torch.tensor(X_test, dtype=torch.float32).to(device=device)
        y_tensor_test = torch.tensor(y_test, dtype=torch.float32).to(device=device) if mode != "classification" else \
                    torch.tensor(y_test, dtype=torch.int32).to(device=device)

        feature_names = [f"Feature_{i}" for i in range(X.shape[1])] #Convert to dataframe for invase method
        X_df_train = pd.DataFrame(X_train, columns=feature_names)
        y_series_train = pd.Series(y_train, name="Target")
        X_df_test = pd.DataFrame(X_test, columns=feature_names)
        


        # Apply each feature selector
        for selector in feature_selectors:
            print(f"Applying feature selector: {selector} on dataset: {dataset_name}")
           

            L2X_explainer, _ = train_L2X(X_tensor_train, y_tensor_train, d, epochs= epoch , batch_size = 200)
            model_filename = f"trained_models/L2X_{dataset_name}.pkl"
            with open(model_filename, "wb") as f:
                    pickle.dump(L2X_explainer, f)
            # L2X_explainer.eval()
            # with torch.no_grad():
            #      _, l_L2X_weights= L2X_explainer(X_tensor_test, training=False)
            # L2X_weights= l_L2X_weights.cpu().numpy()
            # L2X_selected_features = (L2X_weights > 1e-3).astype(int)
            # model_filename_weights = f"importance_weights/L2X_weights_{dataset_name}.pkl"
            # with open(model_filename_weights, "wb") as f:
            #         pickle.dump(L2X_weights, f)
            # del L2X_explainer
            # memory_cleaning()


            # Invase_explainer = INVASE (model, X_df_train, y_series_train, n_epoch=epoch, prefit=False).to(device=device) #prefit = False to train the model
            # model_filename = f"trained_models/INVASE_{dataset_name}.pkl"
            # with open(model_filename, "wb") as f:
            #         pickle.dump(Invase_explainer, f)
            # invase_scores =(Invase_explainer.explain(X_df_test)).to_numpy()                      
            # invase_selected_features = (invase_scores > 0.5).astype(int)
            # model_filename_weights = f"importance_weights/INVASE_weights_{dataset_name}.pkl"
            # with open(model_filename_weights, "wb") as f:
            #         pickle.dump(invase_scores, f)
            # del Invase_explainer
            # memory_cleaning()
           

    print("All datasets processed!")
    