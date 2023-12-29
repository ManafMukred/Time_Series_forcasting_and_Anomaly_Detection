from lightgbm import LGBMRegressor
from DataProcessor import DataProcessor
from Trainer import Trainer, get_MAPE, get_SMAPE
from AnomalyDetector import AnomalyDetector

# choose ML model for time series
MODEL =  LGBMRegressor(random_state=42)
PARAM_GRID = {
            'num_leaves': [16, 24, 31],
            'learning_rate': [0.005, 0.01, 0.05],
            'n_estimators': [32, 100],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            }

if __name__ == "__main__":

    Turbine = DataProcessor('Turbine1.csv', 'Turbine2.csv')
    
    # Clean and reorder data for Turbine1 and Turbine2
    Turbine.df1 = Turbine.clean(Turbine.df1)
    Turbine.df2 = Turbine.clean(Turbine.df2)
    
    # Create additional time series features
    Turbine.df1 = Turbine.add_features(Turbine.df1)
    Turbine.df2 = Turbine.add_features(Turbine.df2)
    
    # Combine and aggregate the data
    combined_data = Turbine.aggregate()
    
    # initiate Trainer class
    trainer = Trainer(combined_data)

    # Auto feature selection
    trainer.features = trainer.select_features(n= 10) #Or set it manually e.g: ["Wind"]   
    
    # Split data into train and test sets
    x_train, x_test, y_train, y_test = trainer.data_splitter()

    # fine tune via GridSearch
    MODEL = trainer.tune(PARAM_GRID, MODEL, x_train, y_train)
    # Train and evaluate the model
    lgb_model, MAE_scores = trainer.train(MODEL, x_train, y_train)
    sampe, mape, mae = trainer.evaluate(lgb_model, x_test, y_test)
    print('-' * 50)
    print('cross validated MAE: %.3f ' % (MAE_scores.mean()))
    print('-' * 50)
    print("Test set MAPE:", mape)
    print("Test set SMAPE:", sampe)
    print("Test set MAE:", mae)
    

    # Load your combined_df data here (assuming it's a DataFrame)
    # combined_df = ...

    # Instantiate AnomalyDetector
    anomaly_detector = AnomalyDetector(Turbine.df1)
    # Fit the Isolation Forest model
    anomaly_detector.fit_model()
    # Detect anomalies
    anomaly_detector.detect_anomalies()
    # Visualize anomalies
    anomaly_detector.visualize_anomalies(until_date= '2016-03-31')
    # show t-SNE
    anomaly_detector.visualize_tsne()