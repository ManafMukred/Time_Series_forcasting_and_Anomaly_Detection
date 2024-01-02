import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import make_scorer, mean_absolute_error, mean_absolute_percentage_error

class Trainer():
    """
    A class for training process and evaluation
    """

    def __init__(self, data):
        """
        Initialize the instance of Trainer

        Creates class variables and retrieves updated data from DataProcessor class 
        """
        self.data = data
        self.features = []
        self.target = 'Leistung'

    def select_features(self, n= 10):
        """
        Select the n best features 

        Parameters:
        - n (int): the number of features to select

        Returns:
        A list with best feature names
        """
        best_features=[]
        X = self.data[self.data.columns.difference([self.target, 'Dat/Zeit'])]
        Y = self.data[self.target]
        # configure to select all features
        fs = SelectKBest(score_func=mutual_info_regression, k='all')
        # learn relationship from training data
        fs.fit(X, Y)
        # get column indecies of best features
        col_idxs = np.argsort(fs.scores_)[-(n):]
        for i in  col_idxs:
            best_features.append(X.iloc[:, i].name)
        return best_features  
    
    def data_splitter(self, z_score=True, pct=0.2):
        """
        Split and standardize data 

        Parameters:
        - pct (float): percentage of test set
        - z_score (bool): make z-score or not

        Returns:
        train_test_split function with four tuples: x_train, x_test, y_train, y_test 
        """
        X = self.data[self.features]
        Y = self.data[self.target]
        if  z_score:
            # Check for zero variance
            non_zero_var_columns = X.columns[X.var() != 0]
            X = X[non_zero_var_columns]
            X_zscore = zscore(X)
            return train_test_split(X_zscore, Y, test_size= pct, shuffle=False)

        else:
            return train_test_split(X, Y, test_size=0.2, shuffle=False)    
 
    def tune(self, param, model, x_train, y_train):
        """
        Search for best model parameters

        Parameters:
        - param (dict): dictionary with parameters and their values to search
        - model : the model class object
        - x_train : train set of feature columns
        - y_train : train set of target column

        Returns:
        Model object with best parameters
        """
        # Create a time series split cross-validator
        tscv = TimeSeriesSplit(n_splits=5)

        # Define the scoring metric (use a custom scorer if needed)
        scoring = make_scorer(mean_absolute_error, greater_is_better=False)

        # Create GridSearchCV object
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param,
            scoring=scoring,
            cv=tscv,
            verbose=1,
            n_jobs=-1
        )

        # Fit the GridSearchCV object to the data
        grid_search.fit(x_train, y_train)

        # Get the best model
        return grid_search.best_estimator_

    def train(self, model, x_train, y_train):
        """
        Train the data

        Parameters:
        - model : the model class object
        - x_train : train set of feature columns
        - y_train : train set of target column

        Returns:
        - Trained model
        - cross validated score on training set (mean_absolute_error)
        """
        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(model, x_train, y_train, scoring=make_scorer(mean_absolute_error), cv=tscv, n_jobs=-1)
        scores = np.absolute(scores)
        model.fit(x_train, y_train)
        return model, scores

    def evaluate(self, model, x_test, y_test):
        """
        Evaluate the data

        Parameters:
        - model : the model class object
        - x_test : test set of feature columns
        - y_test : test set of target column

        Returns:
        - Symmetric mean absolute percentage error (SMAPE)
        - Mean absolute percentage error (MAPE)
        - Mean absolute error (MAE)
        """
        y_pred = model.predict(x_test)
        mape = get_MAPE(y_test, y_pred)
        smape = get_SMAPE(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        return smape, mape, mae    


def get_MAPE(true, pred):
    """
    Calculate  Mean absolute percentage error (SMAPE)

    Returns:
    MAPE score
    """
    non_zero_indices = true.values != 0
    true, pred = true[non_zero_indices], pred[non_zero_indices]
    return mean_absolute_percentage_error(true, pred)

def get_SMAPE(true, pred):
    """
    Calculate Symmetric mean absolute percentage error (MAPE)

    Returns:
    SMAPE score
    """
    return 2 * np.mean(np.abs(true - pred) / (np.abs(true) + np.abs(pred)))