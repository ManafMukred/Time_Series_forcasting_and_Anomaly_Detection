import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import make_scorer, mean_absolute_error, mean_absolute_percentage_error

class Trainer():

    def __init__(self, data):
#         super().__init__(parent.df1, parent.df2)
        self.data = data
        self.features = []
        self.target = 'Leistung'

    def select_features(self, n= 10):
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

    def get_Xval_score(self, model, X, Y, n_splits=10):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = cross_val_score(model, X, Y, scoring=make_scorer(mean_absolute_error), cv=tscv, n_jobs=-1)
        scores = np.absolute(scores)
        print('MAE: %.3f standard deviation: %.3f' % (scores.mean(), scores.std()))

    def tune(self, param, model, x_train, y_train):
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
        model.fit(x_train, y_train)
        return model

    def evaluate(self, model, x_test, y_test):
        y_pred = model.predict(x_test)
        mape = get_MAPE(y_test, y_pred)
        smape = get_SMAPE(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        return smape, mape, mae    


def get_MAPE(true, pred):
    non_zero_indices = true.values != 0
    true, pred = true[non_zero_indices], pred[non_zero_indices]
    return mean_absolute_percentage_error(true, pred)

def get_SMAPE(true, pred):
    return 2 * np.mean(np.abs(true - pred) / (np.abs(true) + np.abs(pred)))