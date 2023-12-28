import pytest
import pandas as pd
import numpy as np
from DataProcessor import DataProcessor
from Trainer import Trainer, get_MAPE, get_SMAPE
from AnomalyDetector import AnomalyDetector
from lightgbm import LGBMRegressor


@pytest.fixture
def random_data_processor():
    yield DataProcessor('Turbine1.csv', 'Turbine2.csv')

def test_clean_function(random_data_processor):

    random_data_processor.df1 = random_data_processor.clean(random_data_processor.df1)
    random_data_processor.df2 = random_data_processor.clean(random_data_processor.df2)
    # Perform assertions
    assert len(random_data_processor.df1) == 13103  # Assuming the last row is removed
    assert 'BtrStd 1' not in random_data_processor.df1.columns
    assert len(random_data_processor.df2) == 13103  # Assuming the last row is removed
    assert 'BtrStd 1' not in random_data_processor.df2.columns
    # Add more assertions as needed

def test_add_features_function(random_data_processor):
    random_data_processor.df1 = random_data_processor.clean(random_data_processor.df1)
    random_data_processor.df2 = random_data_processor.clean(random_data_processor.df2)
    random_data_processor.df1 = random_data_processor.add_features(random_data_processor.df1)
    random_data_processor.df2 = random_data_processor.add_features(random_data_processor.df2)

    # Perform assertions
    assert 'hour' in random_data_processor.df1.columns
    assert 'dayofweek' in random_data_processor.df2.columns
    assert 'hour' in random_data_processor.df1.columns
    assert 'dayofweek' in random_data_processor.df2.columns
    # Add more assertions as needed



@pytest.fixture
def random_trainer(random_data_processor):
    random_data_processor.df1 = random_data_processor.clean(random_data_processor.df1)
    random_data_processor.df2 = random_data_processor.clean(random_data_processor.df2)
    random_data_processor.df1 = random_data_processor.add_features(random_data_processor.df1)
    random_data_processor.df2 = random_data_processor.add_features(random_data_processor.df2)
    return Trainer(random_data_processor.aggregate())

def test_select_features_function(random_trainer):
    random_trainer.features = random_trainer.select_features(n=7)
    # Perform assertions
    assert len(random_trainer.features) == 7
    # Add more assertions as needed

def test_data_splitter_function(random_trainer):
    random_trainer.features = random_trainer.select_features(n=7)
    x_train, x_test, y_train, y_test = random_trainer.data_splitter()

    # Perform assertions
    assert len(x_train) > 0
    assert len(x_test) > 0
    # Add more assertions as needed

# def test_get_Xval_score_function(random_trainer):
#     model = LGBMRegressor(random_state=42)
#     x_train, x_test, y_train, y_test = random_trainer.data_splitter()
#     random_trainer.get_Xval_score(model, x_train, y_train, n_splits=5)

    # Perform assertions (based on printed output, if applicable)
    # Add more assertions as needed

def test_tune_function(random_trainer):
    random_trainer.features = random_trainer.select_features(n=7)
    x_train, x_test, y_train, y_test = random_trainer.data_splitter()
    model = LGBMRegressor(random_state=42)
    x_train, x_test, y_train, y_test = random_trainer.data_splitter()
    param_grid = {'n_estimators': [50, 100, 200]}
    best_model = random_trainer.tune(param_grid, model, x_train, y_train)

    # Perform assertions
    assert best_model is not None
    # Add more assertions as needed

def test_train_and_evaluate_functions(random_trainer):
    random_trainer.features = random_trainer.select_features(n=7)
    x_train, x_test, y_train, y_test = random_trainer.data_splitter()
    model = LGBMRegressor(random_state=42)
    x_train, x_test, y_train, y_test = random_trainer.data_splitter()
    param_grid = {'n_estimators': [50, 100, 200]}
    best_model = random_trainer.tune(param_grid, model, x_train, y_train)
    trained_model = random_trainer.train(best_model, x_train, y_train)
    smape, mape, mae = random_trainer.evaluate(trained_model, x_test, y_test)

    # Perform assertions
    assert mape is not None
    assert smape is not None
    assert mae is not None
    # Add more assertions as needed

def test_get_MAPE_function():
    true_values = pd.Series(np.random.rand(100))
    pred_values = pd.Series(np.random.rand(100))

    mape = get_MAPE(true_values, pred_values)

    # Perform assertions
    assert mape is not None
    # Add more assertions as needed

def test_get_SMAPE_function():
    true_values = pd.Series(np.random.rand(100))
    pred_values = pd.Series(np.random.rand(100))

    smape = get_SMAPE(true_values, pred_values)

    # Perform assertions
    assert smape is not None
    # Add more assertions as needed

@pytest.fixture()
def random_anomaly_detector(random_data_processor):
    random_data_processor.df1 = random_data_processor.clean(random_data_processor.df1)
    random_data_processor.df2 = random_data_processor.clean(random_data_processor.df2)
    random_data_processor.df1 = random_data_processor.add_features(random_data_processor.df1)
    random_data_processor.df2 = random_data_processor.add_features(random_data_processor.df2)
    return AnomalyDetector(random_data_processor.aggregate())

def test_fit_model_function(random_anomaly_detector):
    random_anomaly_detector.fit_model()

    # Perform assertions
    assert random_anomaly_detector.model is not None
    # Add more assertions as needed

def test_detect_anomalies_function(random_anomaly_detector):
    random_anomaly_detector.fit_model()
    updated_data = random_anomaly_detector.detect_anomalies()

    # Perform assertions
    assert 'anomaly' in updated_data.columns
    # Add more assertions as needed

def test_visualize_tsne_function(random_anomaly_detector):
    until_date = '2016-2-2'
    try:    

            pd.to_datetime(until_date, format='%Y-%m-%d')
            assert isinstance(until_date, str), "Date should be in string format"  # Assert that there is no error
    except Exception as e:
            print(f"Error: {e}")
            assert False  # Raise an assertion error if there is an error
    random_anomaly_detector.fit_model()
    random_anomaly_detector.data = random_anomaly_detector.detect_anomalies()   
    random_anomaly_detector.visualize_anomalies(until_date)

 