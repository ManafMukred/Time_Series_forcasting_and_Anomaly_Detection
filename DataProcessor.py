import pandas as pd

class DataProcessor:
    """
    A class for cleaning and formatting dataframes
    """
    
    def __init__(self, file_turbine1, file_turbine2):
        """
        Initialize an instance of DataFrameProcessor
        Reads both Turbine data
        """
        self.df1 = pd.read_csv(file_turbine1, on_bad_lines='warn', delimiter=';')
        self.df2 = pd.read_csv(file_turbine2, on_bad_lines='warn', delimiter=';')

    def clean(self, df):
        """
        Clean data, format columns to numerical and datetime

        Parameters:
        - df  :  pandas Dataframe

        Returns:
        Cleaned dataframe
        """
        df.columns = [col.replace(',', '') for col in df.columns]
        df.columns = df.columns.str.strip()
        df['KH-Ana-4'] = df['KH-Ana-4'].apply(lambda x: x.replace(',', ''))
        df["Dat/Zeit"] = pd.to_datetime(df['Dat/Zeit'][1:], format="%d.%m.%Y, %H:%M", errors="coerce")
        df = df.iloc[1:].reset_index(drop=True)
        numeric_df = df.iloc[:, 1:].replace(',', '.', regex=True).apply(pd.to_numeric)
        df = pd.concat([df['Dat/Zeit'], numeric_df], axis=1)
        df = df.drop(columns=['BtrStd 1', 'BtrStd 2'])
        return df

    def add_features(self, df):
        """
        Add datetime-based features

        Parameters:
        - df  :  pandas Dataframe

        Returns:
        Dataframe with updated fetures
        """
        df['hour'] = df['Dat/Zeit'].dt.hour.astype("int64")
        df['dayofweek'] = df['Dat/Zeit'].dt.dayofweek.astype("int64")
        df['month'] = df['Dat/Zeit'].dt.month.astype("int64")
        df['week'] = df['Dat/Zeit'].dt.isocalendar().week.astype("int64")
        return df

    def aggregate(self):
        """
        combine dataframes and take theeir average

        Returns:
        Combined dataframe 
        """
        combined_df = pd.concat([self.df1, self.df2])
        combined_df = combined_df.groupby(combined_df['Dat/Zeit']).mean()
        return combined_df
