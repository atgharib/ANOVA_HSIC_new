import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from invase import INVASE

class InvaseFeatureImportance:
    def __init__(self, n_epoch=1000):
        self.n_epoch = n_epoch
        self.model = None
        self.explainer = None

    def fit_model(self, X_df, y_series) -> None:
        # Convert NumPy arrays to Pandas DataFrame and Series
        # feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        # X_df = pd.DataFrame(X, columns=feature_names)
        # y_series = pd.Series(y, name="Target")

   
        self.model = LinearRegression()
        self.model.fit(X_df, y_series)

        # Instantiate and fit INVASE explainer
        self.explainer = INVASE(
            self.model, 
            X_df, 
            y_series, 
            n_epoch=self.n_epoch, 
            prefit=True  # The model is already trained
        )

        # Store the training data for later use in the INVASE explainer
        self.X_df = X_df
        self.y_series = y_series

       
    def compute_feature_importance(self, Xt_df):
        """
        Returns:
        - feature_importance: np.ndarray : Feature importance scores as a NumPy array.
        """
        
        # Explain the feature importance for the entire dataset
        explanation = self.explainer.explain(Xt_df)

        return explanation.to_numpy()
