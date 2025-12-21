import numpy as np
import pandas as pd

class LeukemiaClassifier:
    """
    Klasyfikator oparty na odległości euklidesowej od centroidów klas.
    Nie używa sieci neuronowych, lecz czystej statystyki geometrycznej.
    """
    def __init__(self):
        self.means = {}
        self.stds = {}
        self.centroids = {}
        # Cechy do diagnozy
        self.features_to_use = [
        'solidity', 'circularity', 
        'aspect_ratio',
        'color_heterogeneity', 
        'skewness',             
        'texture_std',         
        'std_saturation',       
        'mean_value',
        'nc_ratio',      
        'hu_moment_0'
        ]

    def fit(self, df):
       
        for col in self.features_to_use:
            self.means[col] = df[col].median()
            
            q75 = df[col].quantile(0.75)
            q25 = df[col].quantile(0.25)
            self.stds[col] = q75 - q25
            
            if self.stds[col] == 0: self.stds[col] = 1

        df_norm = self._normalize(df)
        
        self.centroids = df_norm.groupby('label')[self.features_to_use].median().to_dict('index')
        
        clean_indices = []
        
        
        for label, group in df_norm.groupby('label'):
            
            centroid = self.centroids[label]
            
            
            distances = []
            for idx, row in group.iterrows():
                dist = 0
                for col in self.features_to_use:
                    dist += (row[col] - centroid[col]) ** 2
                distances.append(np.sqrt(dist))
            
            distances = np.array(distances)
            
            limit = np.percentile(distances, 90) 
            
            good_samples = group[distances <= limit]
            clean_indices.extend(good_samples.index)
            
        
        print(f"Odrzucono {len(df) - len(clean_indices)} wątpliwych przypadków (szum/błędy segmentacji).")
        
        
        df_norm_clean = df_norm.loc[clean_indices]
        
        
        self.centroids = df_norm_clean.groupby('label')[self.features_to_use].median().to_dict('index')
        print(f"Klasyfikator wytrenowany na oczyszczonych danych.")

    def predict(self, df):
        """
        Rozpoznawanie dla całej tabeli danych.
        Zwraca listę predykcji.
        """
        df_norm = self._normalize(df)
        predictions = []

        for _, row in df_norm.iterrows():
            pred = self._predict_single_normalized(row)
            predictions.append(pred)
        
        return predictions

    def _normalize(self, df):
        """Pomocnicza funkcja do skalowania danych."""
        df_copy = df.copy()
        for col in self.features_to_use:
            df_copy[col] = (df_copy[col] - self.means[col]) / self.stds[col]
        return df_copy

    def _predict_single_normalized(self, row):
        weights = {
            'solidity': 1.5,
            'circularity': 1.0,
            'color_heterogeneity': 2.0, 
            'skewness': 2.0,            
            'texture_std': 1.5,
            'aspect_ratio': 0.5,        
            'std_saturation': 1.5,
            'mean_value': 1.0,
            'nc_ratio': 2.5,
            'hu_moment_0': 1.2
        }
        
        best_label = None
        min_dist = float('inf')

        for label, centroid in self.centroids.items():
            dist = 0
            for feature in self.features_to_use:
                w = weights.get(feature, 1.0)
                dist += w * ((row[feature] - centroid[feature]) ** 2)
            
            
            if dist < min_dist:
                min_dist = dist
                best_label = label
        
        return best_label