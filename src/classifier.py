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
        # Cechy, które bierzemy pod uwagę przy decyzji (kluczowe dla diagnozy)
        self.features_to_use = [
        'solidity', 'circularity', 
        'aspect_ratio',
        'color_heterogeneity', 
        'skewness',             
        'texture_std',         
        'std_saturation',       
        'mean_value'
        ]

    def fit(self, df):
        # 1. Oblicz statystyki używając MEDIANY i IQR (rozstęp międzykwartylowy) zamiast średniej/std
        for col in self.features_to_use:
            self.means[col] = df[col].median()  # Mediana zamiast średniej!
            
            # IQR zamiast odchylenia standardowego
            q75 = df[col].quantile(0.75)
            q25 = df[col].quantile(0.25)
            self.stds[col] = q75 - q25
            
            if self.stds[col] == 0: self.stds[col] = 1

        df_norm = self._normalize(df)

        # 2. Centroidy też wyznaczaj medianą (Centroid staje się "Medoidem")
        self.centroids = df_norm.groupby('label')[self.features_to_use].median().to_dict('index')
        print(f"Klasyfikator (oparty na medianie) wytrenowany.")

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
            'mean_value': 1.0
        }
        
        best_label = None
        min_dist = float('inf')

        for label, centroid in self.centroids.items():
            dist = 0
            for feature in self.features_to_use:
                w = weights.get(feature, 1.0)
                # Dystans ważony
                dist += w * ((row[feature] - centroid[feature]) ** 2)
            
            # Pierwiastek nie jest konieczny do porównywania (monotoniczność), ale można zostawić
            if dist < min_dist:
                min_dist = dist
                best_label = label
        
        return best_label