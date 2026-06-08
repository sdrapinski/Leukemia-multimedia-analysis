# Klasyczna analiza obrazu a głębokie uczenie w wykrywaniu białaczki na obrazach mikroskopowych komórek krwi — analiza porównawcza

**Przedmiot:** Analiza danych multimedialnych
**Prowadzący:** Łukasz Jeleń
**Autorzy:** Szymon Drapiński, Maciej Adamski, Dominik Ochej, Jakub Cegłowski
**Grupa:** SliUM2m G

---

## Streszczenie

W pracy porównano dwa podejścia do automatycznego rozróżniania zdrowych białych krwinek (`hem`) od komórek dotkniętych ostrą białaczką limfoblastyczną (`all`) na obrazach mikroskopowych. Podejście pierwsze (semestr poprzedni) opiera się na **klasycznym potoku przetwarzania obrazu**: segmentacji komórki, ekstrakcji ręcznie projektowanych cech morfologicznych, kolorystycznych i teksturalnych oraz klasyfikacji metodą ważonego najbliższego centroidu. Podejście drugie (semestr bieżący) wykorzystuje **głębokie uczenie** — konwolucyjną sieć neuronową VGG16 w trybie transfer learning, uczącą reprezentację bezpośrednio z pikseli. Pokazujemy, że obie metody mają komplementarne mocne i słabe strony, a bezpośrednie porównanie samej dokładności jest mylące ze względu na różnice w skali i trudności zbiorów danych oraz w rygorze ewaluacji.

**Słowa kluczowe:** białaczka, ALL, klasyfikacja obrazów medycznych, cechy ręcznie projektowane, segmentacja, CNN, VGG16, transfer learning, augmentacja danych.

---

## 1. Wprowadzenie

Celem obu projektów jest wspomaganie diagnostyki białaczki przez automatyczną klasyfikację zdjęć mikroskopowych pojedynczych białych krwinek na dwie klasy: komórki **zdrowe** (`hem`) i **nowotworowe** (`all`). Z perspektywy medycznej najkosztowniejszym błędem jest **wynik fałszywie negatywny** — uznanie komórki chorej za zdrową — dlatego oprócz ogólnej dokładności kluczowy jest *recall* klasy `all`.

Pierwsza wersja projektu (plik `main.ipynb`) realizuje klasyczny potok widzenia komputerowego, w którym ekspert ręcznie definiuje cechy różnicujące komórki. Druga wersja (plik `mainCnn.ipynb`) zastępuje ręczną inżynierię cech automatycznym uczeniem reprezentacji przez sieć konwolucyjną. Niniejszy artykuł zestawia oba rozwiązania pod kątem metodyki, wyników i przydatności praktycznej.

---

## 2. Dane

Oba podejścia korzystają z tego samego źródła (zbiór **C-NMC Leukemia**), lecz w różnym zakresie i organizacji.

| | Metoda klasyczna (`main.ipynb`) | Metoda CNN (`mainCnn.ipynb`) |
|---|---|---|
| Katalog | `data/v1` | `data/fold_0`, `data/fold_2` |
| Liczba obrazów | **114** (65 `healthy` + 49 `cancer`) | trening **4 794**, walidacja **4 914** |
| Podział | losowy 80% / 20% (ten sam rozkład) | **cross-fold**: trening = fold_0, walidacja = fold_2 |
| Rozdzielczość wejścia | natywna (450×450) | skalowane do 128×128 |
| Balans klas | ~57% / 43% | wyrównany augmentacją do **1:1** |

> **Uwaga metodyczna.** W wersji CNN klasa mniejszościowa `hem` została rozszerzona offline'ową augmentacją (`utils/augment_hem.py`) do liczebności klasy `all`, zarówno w foldzie treningowym, jak i walidacyjnym. Dzięki temu zbiór treningowy jest zbalansowany (`class_weight = {0: 1.0, 1: 1.0}`), ale rozkład klas w walidacji zmienił się z naturalnego (~31% `hem`) na sztuczny 50/50 — co należy uwzględnić przy interpretacji metryk (sekcja 6).

Zasadnicza różnica: metoda klasyczna pracuje na **małym, wyselekcjonowanym** zbiorze, podczas gdy metoda CNN — na **dużym, znacznie trudniejszym** zbiorze z walidacją międzyfoldową (inne preparaty w treningu i walidacji).

---

## 3. Metoda I — klasyczny potok przetwarzania obrazu

Potok (`main.ipynb` + klasy z `src/`) składa się z czterech etapów:

1. **Przetwarzanie wstępne** — konwersja do skali szarości, poprawa kontrastu metodą **CLAHE**.
2. **Segmentacja** — maska komórki wyznaczana z **mapy entropii lokalnej** progowanej metodą Otsu, oczyszczana operacjami morfologicznymi (zamknięcie + otwarcie).
3. **Ekstrakcja cech** (`FeatureExtractor`) z obszaru maski:
   - *morfologiczne:* pole, obwód, **kołowość**, **solidność**, proporcje (aspect ratio), **momenty Hu**;
   - *kolorystyczne:* średnie/odchylenia HSV, **heterogeniczność koloru** (Lab), odchylenie nasycenia;
   - *teksturalne:* odchylenie jasności, różnica max-średnia, **skośność** rozkładu jasności;
   - *strukturalne:* **stosunek jądro/cytoplazma** (`nc_ratio`) szacowany progowaniem Otsu wewnątrz maski.
4. **Klasyfikacja** (`LeukemiaClassifier`) — **ważony najbliższy centroid**: cechy normalizowane robustowo (mediana i rozstęp międzykwartylowy IQR), centroidy klas liczone jako mediany, z odrzuceniem 10% najbardziej odstających próbek w treningu; w predykcji odległość euklidesowa ważona ręcznie dobranymi wagami cech (największe dla `nc_ratio`, `color_heterogeneity`, `skewness`).

Cechą charakterystyczną tego podejścia jest pełna **interpretowalność** — każda decyzja wynika ze zrozumiałych, mierzalnych właściwości komórki.

---

## 4. Metoda II — głębokie uczenie (VGG16, transfer learning)

Potok (`mainCnn.ipynb`):

1. **Normalizacja barwienia** — **Reinhard stain normalization** (dopasowanie średniej i odchylenia w przestrzeni LAB do obrazu referencyjnego) + **CLAHE na kanale L**, by sieć nie uczyła się różnic w barwieniu preparatów jako cechy klasy.
2. **Augmentacja** — dwupoziomowa: (a) offline'owe wyrównanie klasy `hem` (odbicia, rotacje 0–360°, zoom, translacja, łagodny jitter kontrastu/gamma), (b) warstwy `RandomFlip/Rotation/Zoom/Translation` aktywne w treningu.
3. **Sieć** — baza **VGG16** pretrenowana na ImageNet (bez ostatniej warstwy) + głowa: `GlobalAveragePooling → BatchNorm → Dropout(0.5) → Dense(128, ReLU) → BatchNorm → Dropout(0.3) → Dense(1, sigmoid)`.
4. **Trening dwuetapowy:** Stage 1 — warm-up głowy (baza zamrożona, LR=1e-3); Stage 2 — fine-tuning bloku `block5` VGG16 (LR=1e-5). Callbacki: EarlyStopping (`val_auc`), ReduceLROnPlateau, ModelCheckpoint.
5. **Dobór progu** — optymalizacja progu decyzyjnego pod **macro-F1** zamiast domyślnego 0.5.

Cechą charakterystyczną jest automatyczne **uczenie reprezentacji** z pikseli — bez segmentacji i bez ręcznego definiowania cech — kosztem interpretowalności i mocy obliczeniowej.

---

## 5. Wyniki

### 5.1. Metoda klasyczna

Na zbiorze testowym (20% z 114 obrazów, po odrzuceniu 10 wątpliwych przypadków w treningu):

| Metryka | Wartość |
|---|---|
| **Dokładność (accuracy)** | **91,30%** |
| Komórki zdrowe (`healthy`) | rozpoznane bezbłędnie |
| Błędy | 2 przypadki `cancer` → `healthy` (fałszywie negatywne) |

Błędy dotyczyły komórek nowotworowych z **subtelnymi** zmianami (wysoka solidność ~0,97–0,99, niska heterogeniczność koloru), morfologicznie zbliżonych do zdrowych.

### 5.2. Metoda CNN (walidacja na foldzie 2, 4 914 obrazów)

Metryki niezależne od progu:

| Metryka | Wartość |
|---|---|
| **ROC AUC** | **0,7705** |
| **PR AUC** | 0,7073 |
| Optymalny próg (macro-F1) | 0,56 |

Raport klasyfikacji dla progu domyślnego **0,50**:

| Klasa | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Zdrowe (`hem`) | 0,820 | 0,545 | 0,655 | 2 457 |
| Białaczka (`all`) | 0,659 | 0,880 | 0,754 | 2 457 |
| **macro avg** | 0,740 | 0,713 | 0,705 | 4 914 |
| **accuracy** | | | **0,713** | 4 914 |

Raport dla progu zoptymalizowanego **0,56**:

| Klasa | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Zdrowe (`hem`) | 0,788 | 0,583 | 0,670 | 2 457 |
| Białaczka (`all`) | 0,669 | 0,843 | 0,746 | 2 457 |
| **macro avg** | 0,728 | 0,713 | 0,708 | 4 914 |
| **accuracy** | | | **0,713** | 4 914 |

Wykresy diagnostyczne (zapisane przez notebook):

- Krzywe uczenia: `output/training_curves.png`
- Krzywe ROC i Precision-Recall: `output/roc_pr_curves.png`
- Dobór progu (macro-F1): `output/threshold_tuning.png`
- Macierze pomyłek: `output/confusion_matrices.png`

---

## 6. Analiza porównawcza

### 6.1. Zestawienie zbiorcze

| Aspekt | Metoda klasyczna | Metoda CNN (VGG16) |
|---|---|---|
| Reprezentacja cech | ręcznie projektowana | uczona automatycznie |
| Segmentacja komórki | wymagana | niepotrzebna |
| Normalizacja barwy | CLAHE (szarość) | Reinhard + CLAHE (kanał L) |
| Augmentacja danych | brak | offline (1:1) + warstwy Keras |
| Klasyfikator | ważony najbliższy centroid | VGG16 + głowa MLP (sigmoid) |
| Wielkość zbioru | 114 obrazów | ~9 700 obrazów (2 foldy) |
| Ewaluacja | split 80/20 (ten sam rozkład) | cross-fold (inne preparaty) |
| **Dokładność** | **91,3%** | **71,3%** |
| Interpretowalność | **wysoka** | niska (black-box) |
| Koszt obliczeniowy | niski (sekundy, CPU) | wysoki (~2 h treningu, CPU) |
| Skalowalność | ograniczona (zależy od segmentacji) | **wysoka** |

### 6.2. Dlaczego CNN ma niższą dokładność? — porównanie nie jest „jeden do jednego"

Pozornie paradoksalny wynik (91,3% vs 71,3%) **nie oznacza wyższości metody klasycznej**. Wynika on z fundamentalnych różnic warunków eksperymentu:

1. **Trudność i skala zbioru.** Metoda klasyczna testowana była na ~23 obrazach z małego, wyselekcjonowanego zbioru `v1` (a w treningu dodatkowo odrzucono 10% najbardziej odstających próbek). CNN oceniono na **4 914** obrazach z pełnego, znacznie bardziej zaszumionego i zróżnicowanego zbioru C-NMC.
2. **Rygor ewaluacji.** Metoda klasyczna używa losowego podziału tego samego rozkładu (trening i test pochodzą z tej samej puli). CNN walidowano **międzyfoldowo** — trenowano na foldzie 0, a oceniano na foldzie 2 (inne preparaty/pacjenci), co jest znacznie surowszym i bardziej realistycznym testem generalizacji.
3. **Wariancja estymaty.** Dokładność liczona na 23 obrazach ma ogromny rozrzut (jeden błąd to ~4 punkty procentowe) i jest optymistycznie obciążona; metryka na ~5 tys. obrazów jest stabilna i wiarygodna.
4. **Zależność od segmentacji.** Skuteczność metody klasycznej stoi i upada na jakości maski. Na czystych, dobrze odseparowanych komórkach `v1` segmentacja działa znakomicie; na pełnym zbiorze odsetek błędów segmentacji byłby znacznie wyższy.

### 6.3. Charakter błędów (perspektywa medyczna)

- **Metoda klasyczna:** błędy to wyłącznie **fałszywie negatywne** (`cancer` → `healthy`) — komórki chore o subtelnych zmianach. To najkosztowniejszy diagnostycznie typ błędu.
- **Metoda CNN:** przy progu 0,50 osiąga **recall klasy `all` = 0,88** (wykrywa 88% komórek chorych), a większość pomyłek to nadrozpoznawanie białaczki (zdrowe oznaczane jako chore) — błąd mniej groźny klinicznie (fałszywy alarm), choć obniżający precyzję dla `hem`.

### 6.4. Uwagi metodyczne i ograniczenia

- **Augmentacja zbioru walidacyjnego.** Wyrównanie klasy `hem` zastosowano także w foldzie walidacyjnym, przez co metryki liczone są na sztucznym rozkładzie 50/50. Dla rzetelnej oceny zaleca się ewaluację na **naturalnym, niezmienionym** rozkładzie (oraz na zarezerwowanym foldzie 1 jako zbiorze testowym).
- **Brak wspólnego benchmarku.** Obie metody nie zostały przetestowane na identycznym zbiorze testowym — to główne ograniczenie porównania. Pełni rzetelne zestawienie wymagałoby uruchomienia metody klasycznej na foldach C-NMC.

---

## 7. Wnioski

1. **Metoda klasyczna** jest interpretowalna, szybka i bardzo skuteczna na małym, czystym zbiorze (91,3%), ale słabo skaluje się i jest krucha wobec błędów segmentacji oraz różnorodności danych rzeczywistych.
2. **Metoda CNN** nie wymaga segmentacji ani ręcznej inżynierii cech, naturalnie skaluje się do dużych zbiorów i została zweryfikowana w surowym scenariuszu międzyfoldowym (ROC AUC = 0,77), lecz na obecnym etapie osiąga niższą dokładność i pozostaje „czarną skrzynką" o wysokim koszcie obliczeniowym.
3. **Niższa dokładność CNN nie świadczy o gorszej metodzie** — wynika z trudniejszego zbioru i bardziej rygorystycznej ewaluacji. To podejście klasyczne działało w warunkach „łatwiejszych".
4. **Rekomendacja:** kierunkiem rozwojowym jest podejście głębokie, ale wymaga ono dalszej pracy: połączenia foldów `fold_0 + fold_1` w większy zbiór treningowy, ewaluacji na naturalnym rozkładzie klas, eksperymentów z nowszymi architekturami (EfficientNet, ResNet) oraz funkcją straty typu *focal loss*. Cechy ręcznie projektowane mogą posłużyć jako interpretowalny punkt odniesienia i element walidacji.

---

## Bibliografia

- [Zbiór danych C-NMC Leukemia (Kaggle)](https://www.kaggle.com/datasets/avk256/cnmc-leukemia/data)
- [VGG16 — Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- E. Reinhard i in., *Color Transfer between Images*, IEEE CG&A, 2001.
- [Dokumentacja TensorFlow / Keras](https://www.tensorflow.org/api_docs)
- [Dokumentacja scikit-learn](https://scikit-learn.org/stable/documentation.html)
- [Dokumentacja OpenCV](https://docs.opencv.org/) · [scikit-image](https://scikit-image.org/docs/stable/)
