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
            'solidity',             # Nieregularność brzegu
            'circularity',          # Okrągłość
            'color_heterogeneity',  # Przebarwienia (Std Lab)
            'aspect_ratio'          # Wydłużenie
        ]

    def fit(self, df):
        """
        'Uczenie' klasyfikatora: obliczenie średnich wartości cech dla zdrowych i chorych.
        """
        # 1. Normalizacja danych (standaryzacja), aby np. 'area' (duże liczby)
        # nie zagłuszyła 'solidity' (małe liczby 0-1).
        for col in self.features_to_use:
            self.means[col] = df[col].mean()
            self.stds[col] = df[col].std()
            if self.stds[col] == 0: self.stds[col] = 1 # Zabezpieczenie przed dzieleniem przez 0

        df_norm = self._normalize(df)

        # 2. Obliczenie centroidów (środków ciężkości) dla każdej klasy
        self.centroids = df_norm.groupby('label')[self.features_to_use].mean().to_dict('index')
        print(f"Klasyfikator wytrenowany. Wzorce klas: {list(self.centroids.keys())}")

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
        """Mierzy odległość punktu od wzorców i zwraca najbliższy."""
        best_label = None
        min_dist = float('inf')

        for label, centroid in self.centroids.items():
            # Obliczanie dystansu euklidesowego w przestrzeni cech
            # d = sqrt((x1-w1)^2 + (x2-w2)^2 + ...)
            dist = 0
            for feature in self.features_to_use:
                dist += (row[feature] - centroid[feature]) ** 2
            dist = np.sqrt(dist)

            if dist < min_dist:
                min_dist = dist
                best_label = label
        
        return best_label