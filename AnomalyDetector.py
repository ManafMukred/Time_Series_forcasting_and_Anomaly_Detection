from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class AnomalyDetector:
    def __init__(self, data, target = 'Leistung'):
        self.model = IsolationForest(random_state=0, contamination=0.15)
        self.data = data
        self.target = target
        
    def fit_model(self):
        """Fit the Isolation Forest model to the specified data."""
        self.model.fit(self.data)

    def detect_anomalies(self):
        """Detect anomalies in the data using the Isolation Forest model."""
        # self.data['score'] = self.model.decision_function(self.data)
        self.data['anomaly'] = self.model.predict(self.data)
        return self.data

    def visualize_anomalies(self, until_date = '' ):
        """Visualize anomalies in the data."""
             
        if not until_date:
            until_date = self.data.index.strftime('%Y-%m-%d')[-1]
        #### Visualize anomalies in specific period
        plt.figure(figsize=(12, 6))  
        
        plt.plot(self.data.index[self.data.index < until_date], 
                 self.data.loc[self.data.index < until_date,self.target],
                 label='Normal', color='blue', linewidth=0.2)

        # Mark outliers with 'o'
        plt.scatter(self.data.index[(self.data['anomaly'] == -1) & (self.data.index < until_date)], 
                    self.data.loc[(self.data['anomaly'] == -1)& (self.data.index < until_date), self.target],
                    c='red', marker='x', label='Anomaly', s=10)
        plt.xlabel('Sample Index')
        plt.ylabel('Output')
        plt.title('Isolation Forest Anomaly Detection')
        plt.legend()
        plt.show()

    def visualize_tsne(self):
        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=0)
        tsne_result = tsne.fit_transform(self.data[self.data.columns.difference(['anomaly', 'score'])])

        # Plot t-SNE with colored anomalies
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=self.data['anomaly'], cmap='coolwarm', marker='o', s=5)
        plt.title('t-SNE Plot with Anomaly Coloring')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.colorbar(label='Anomaly')
        plt.show()
