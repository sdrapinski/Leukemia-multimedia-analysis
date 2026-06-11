# Wykrywanie białaczki na obrazach mikroskopowych komórek krwi: klasyczna analiza obrazu kontra konwolucyjna sieć neuronowa
claude --resume e1dd87d3-ceb5-4a4f-8645-c2cf380461f8
**Przedmiot:** Analiza danych multimedialnych
**Prowadzący:** Łukasz Jeleń
**Autorzy:** Szymon Drapiński, Maciej Adamski, Dominik Ochej, Jakub Cegłowski
**Grupa:** SliUM2m G

---

> ### ℹ️ Jak korzystać z tego dokumentu
> W tekście rozmieszczono ramki oznaczone **📷 SCREEN N**. Każda ramka mówi, **co** sfotografować i **gdzie** to znaleźć (numer komórki w notatniku albo gotowy plik PNG w folderze `output/`). Wystarczy zrzucić wskazane wykresy i wstawić je w miejsce ramek. Pełną listę zrzutów zebrano w *Załączniku A* na końcu. Tę ramkę można usunąć przed oddaniem pracy.

---

## Streszczenie

W pracy porównano dwa podejścia do automatycznego rozróżniania zdrowych białych krwinek od komórek dotkniętych ostrą białaczką limfoblastyczną (ALL) na obrazach mikroskopowych. Podejście pierwsze (semestr poprzedni, plik `main.ipynb`) opiera się na **klasycznym potoku przetwarzania obrazu**: segmentacji komórki, ekstrakcji ręcznie zaprojektowanych cech morfologicznych i kolorystycznych oraz klasyfikacji metodą ważonego najbliższego centroidu. Podejście drugie (semestr bieżący, plik `mainCnn.ipynb`) wykorzystuje **głębokie uczenie** — konwolucyjną sieć neuronową VGG16 w trybie transfer learning, uczącą reprezentację bezpośrednio z pikseli. Pokazujemy, że obie metody mają komplementarne mocne i słabe strony, a bezpośrednie zestawienie samej dokładności jest mylące, ponieważ metody testowano na zbiorach o radykalnie różnej skali i trudności oraz przy różnym rygorze ewaluacji. Metoda klasyczna osiąga 91,3% dokładności na małym, wyselekcjonowanym zbiorze 114 obrazów; metoda CNN osiąga ROC AUC = 0,77 na blisko 5 000 obrazów testowanych w surowym scenariuszu międzyfoldowym.

**Słowa kluczowe:** białaczka, ALL, klasyfikacja obrazów medycznych, cechy ręcznie projektowane, segmentacja, CNN, VGG16, transfer learning, augmentacja danych, normalizacja barwienia.

---

## 1. Wprowadzenie i cel pracy

Celem obu projektów jest wspomaganie diagnostyki białaczki przez automatyczną klasyfikację zdjęć mikroskopowych pojedynczych białych krwinek na dwie klasy: komórki **zdrowe** (oznaczane `hem` / `healthy`) i **nowotworowe** (oznaczane `all` / `cancer`). Z perspektywy medycznej najgroźniejszym błędem jest **wynik fałszywie negatywny** — uznanie komórki chorej za zdrową — dlatego oprócz ogólnej dokładności kluczowy jest *recall* (czułość) klasy nowotworowej.

Pierwsza wersja projektu realizuje klasyczny potok widzenia komputerowego, w którym ekspert ręcznie definiuje cechy różnicujące komórki. Druga wersja zastępuje ręczną inżynierię cech automatycznym uczeniem reprezentacji przez sieć konwolucyjną. Niniejszy artykuł zestawia oba rozwiązania pod kątem metodyki, jakości wyników i przydatności praktycznej, a także wyjaśnia, dlaczego prostego porównania „liczba kontra liczba” nie da się tu przeprowadzić uczciwie.

> 📷 **[ SCREEN 1 ]** — *Przykładowe obrazy wejściowe obu klas*
> **Co pokazać:** kilka surowych zdjęć komórek — 2–3 zdrowe i 2–3 nowotworowe, najlepiej obok siebie, żeby czytelnik zobaczył, jak subtelna jest różnica.
> **Gdzie znaleźć:** pliki obrazów w folderach `data/v1/healthy/` i `data/v1/cancer/` (metoda klasyczna) lub `data/fold_0/fold_0/hem/` i `data/fold_0/fold_0/all/` (metoda CNN). Otwórz po kilka i zrób zbiorczy zrzut, ewentualnie sklej w jeden kolaż.

---

## 2. Dane

Oba podejścia korzystają z tego samego źródła (zbiór **C-NMC Leukemia** z platformy Kaggle), lecz w zupełnie innym zakresie i organizacji.

| Cecha zbioru | Metoda klasyczna (`main.ipynb`) | Metoda CNN (`mainCnn.ipynb`) |
|---|---|---|
| Katalog | `data/v1` | `data/fold_0` (trening), `data/fold_2` (walidacja) |
| Liczba obrazów | **114** (65 zdrowych + 49 chorych) | trening **4 794**, walidacja **4 914** |
| Sposób podziału | losowy 80% / 20% z tej samej puli | **międzyfoldowy**: trening = fold_0, walidacja = fold_2 (inne preparaty) |
| Rozdzielczość wejścia | natywna (ok. 450×450 px) | skalowane do 128×128 px |
| Naturalny balans klas | ~57% zdrowe / 43% chore | ~32% zdrowe / 68% chore |
| Balans po przygotowaniu | bez zmian | **wyrównany augmentacją do 1:1** |

**Augmentacja klasy mniejszościowej.** W wersji CNN klasa zdrowa (`hem`) była w obu foldach znacznie mniej liczna od chorej (`all`). Skrypt `utils/augment_hem.py` rozszerzył ją offline (odbicia, rotacje, zoom, translacje) do liczebności klasy `all`. W efekcie:

- **trening (fold_0):** 2 397 `all` + 2 397 `hem` (z czego 1 267 to obrazy sztucznie wygenerowane),
- **walidacja (fold_2):** 2 457 `all` + 2 457 `hem` (z czego 1 361 wygenerowanych; naturalnych zdrowych komórek jest tu tylko 1 096).

Dzięki temu zbiór treningowy jest idealnie zbalansowany, więc wagi klas wyszły neutralne (`class_weight = {0: 1.0, 1: 1.0}`). **Uwaga metodyczna:** ponieważ augmentacją objęto także fold walidacyjny, metryki CNN liczone są na sztucznym rozkładzie 50/50, a nie na rozkładzie naturalnym (~31% zdrowych). To istotne przy interpretacji wyników (wracamy do tego w sekcji 7).

Zasadnicza różnica między zbiorami: metoda klasyczna pracuje na **małym, wyselekcjonowanym** zbiorze czystych komórek, a metoda CNN — na **dużym, znacznie trudniejszym** zbiorze z walidacją na innych preparatach niż te użyte do treningu.

---

## 3. Metoda I — klasyczny potok przetwarzania obrazu

Potok (`main.ipynb` wspierany klasami z katalogu `src/`) składa się z czterech etapów:

1. **Przetwarzanie wstępne** — konwersja obrazu do skali szarości i poprawa kontrastu metodą **CLAHE** (adaptacyjne wyrównywanie histogramu).
2. **Segmentacja** — maska komórki wyznaczana z **mapy entropii lokalnej** progowanej metodą Otsu, a następnie oczyszczana operacjami morfologicznymi (zamknięcie i otwarcie). Pozwala to oddzielić komórkę od tła.
3. **Ekstrakcja cech** (klasa `FeatureExtractor`) z obszaru maski. Cechy dzielą się na grupy:
   - *morfologiczne:* pole, obwód, **kołowość**, **solidność** (wypukłość kształtu), proporcje boków (aspect ratio), momenty Hu;
   - *kolorystyczne:* średnie i odchylenia w przestrzeni HSV, **heterogeniczność koloru** (odchylenie w przestrzeni Lab), odchylenie nasycenia;
   - *teksturalne:* odchylenie jasności, różnica maksimum–średnia, skośność rozkładu jasności;
   - *strukturalne:* **stosunek jądro/cytoplazma** (`nc_ratio`).
4. **Klasyfikacja** (klasa `LeukemiaClassifier`) — **ważony najbliższy centroid**: cechy są normalizowane odpornościowo (mediana i rozstęp międzykwartylowy IQR), centroidy klas liczone jako mediany z odrzuceniem 10% najbardziej odstających próbek treningowych, a w predykcji liczona jest odległość euklidesowa ważona ręcznie dobranymi wagami cech (najwyższe dla `nc_ratio`, heterogeniczności koloru i skośności).

Cechą charakterystyczną tego podejścia jest pełna **interpretowalność** — każda decyzja wynika ze zrozumiałych, mierzalnych właściwości komórki, a wpływ poszczególnych cech można prześledzić na wykresach rozkładów.

> 📷 **[ SCREEN 2 ]** — *Rozróżnialność klas: kształt i tekstura*
> **Co pokazać:** dwa wykresy pudełkowe (boxplot) — kołowość oraz tekstura/ziarnistość — z podziałem na komórki zdrowe i chore.
> **Gdzie znaleźć:** `main.ipynb`, komórka 13 (wykres „Porównanie kształtu (Kołowość)” i „Porównanie tekstury (Ziarnistość)”).

> 📷 **[ SCREEN 3 ]** — *Rozróżnialność klas: anomalie kolorystyczne*
> **Co pokazać:** wykresy niejednolitości koloru (std saturation) oraz obecności jasnych plam (max − mean) oraz histogram solidności / heterogeniczności koloru.
> **Gdzie znaleźć:** `main.ipynb`, komórka 14 (anomalie kolorystyczne) oraz komórka 15 (histogramy solidności i aspect ratio + boxplot heterogeniczności koloru). Można wstawić jako jeden lub dwa zrzuty.

---

## 4. Metoda II — głębokie uczenie (VGG16, transfer learning)

Potok z `mainCnn.ipynb` zachowuje ten sam cel, ale całkowicie zmienia sposób, w jaki komórka jest reprezentowana i klasyfikowana.

1. **Normalizacja barwienia** — **Reinhard stain normalization**: w przestrzeni LAB średnia i odchylenie każdego obrazu są przesuwane do wartości referencyjnych wyliczonych z jednego obrazu treningowego. Dzięki temu sieć nie uczy się różnic w barwieniu preparatów jako cechy klasy. Dodatkowo na kanale L stosujemy **CLAHE**, co uwydatnia strukturę jądra bez utraty informacji barwnej.
2. **Augmentacja danych** — dwupoziomowa: (a) offline'owe wyrównanie liczebności klasy zdrowej (opisane w sekcji 2), (b) warstwy `RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomTranslation` aktywne tylko podczas treningu, dzięki czemu sieć w każdej epoce widzi inny wariant obrazu. Komórki krwi nie mają wyróżnionej orientacji, więc obroty i odbicia są naturalne i bardzo skuteczne.
3. **Sieć** — baza **VGG16** wstępnie wytrenowana na zbiorze ImageNet (bez warstwy klasyfikującej) z dołączoną własną głową: `GlobalAveragePooling → BatchNormalization → Dropout(0,5) → Dense(128, ReLU) → BatchNormalization → Dropout(0,3) → Dense(1, sigmoid)`. Model liczy ok. **14,78 mln** parametrów.
4. **Trening dwuetapowy:**
   - *Stage 1 — rozgrzewka głowy* (LR = 1e‑3): baza VGG16 całkowicie zamrożona, uczy się tylko nowa głowa. Chroni to wytrenowane wagi VGG16 przed zniszczeniem dużymi gradientami z losowo zainicjalizowanej głowy.
   - *Stage 2 — fine-tuning bloku `block5` VGG16* (LR = 1e‑5): odmrażamy ostatni blok konwolucyjny i dostrajamy go z bardzo małym tempem uczenia do specyfiki obrazów mikroskopowych.
   - Callbacki: `EarlyStopping` (monitoruje `val_auc`), `ReduceLROnPlateau`, `ModelCheckpoint` (zapisuje najlepszy model).
5. **Dobór progu decyzyjnego** — zamiast domyślnego progu 0,5 szukamy progu maksymalizującego **macro-F1**, aby poprawić wykrywanie klasy mniejszościowej.

Cechą charakterystyczną jest automatyczne **uczenie reprezentacji** z pikseli — bez segmentacji i bez ręcznego definiowania cech — kosztem interpretowalności i znacznie większego zapotrzebowania na moc obliczeniową.

> 📷 **[ SCREEN 4 ]** — *Etapy przetwarzania obrazu w potoku CNN*
> **Co pokazać:** zbiorczy podgląd kolejnych etapów obróbki (obraz oryginalny → po normalizacji barwienia → po CLAHE → rozbicie na kanały R/G/B) dla kilku przykładowych komórek.
> **Gdzie znaleźć:** gotowy plik **`output/preprocessing_stages/_overview.png`** (najwygodniej) albo `mainCnn.ipynb`, komórka 11 (wyświetlana siatka 36 obrazów).

> 📷 **[ SCREEN 5 ]** — *Architektura modelu (podsumowanie warstw)*
> **Co pokazać:** tabelę `model.summary()` z warstwami i liczbą parametrów (Total params: 14 783 041).
> **Gdzie znaleźć:** `mainCnn.ipynb`, komórka 13 (wydruk „Model: vgg16_leukemia”).

---

## 5. Przebieg treningu sieci CNN

Trening prowadzono na CPU (brak wsparcia GPU dla TensorFlow na natywnym Windows), co przekłada się na czas rzędu ~150 s na epokę. Krzywe uczenia pokazują stabilny wzrost dokładności i AUC oraz moment przejścia z rozgrzewki głowy do fine-tuningu (zaznaczony pionową linią). Brak rozjeżdżania się krzywych treningowej i walidacyjnej świadczy o tym, że augmentacja i dropout skutecznie ograniczyły przeuczenie.

> 📷 **[ SCREEN 6 ]** — *Krzywe uczenia (dokładność, strata, AUC)*
> **Co pokazać:** trzy wykresy przebiegu treningu z czerwoną linią rozdzielającą Stage 1 i Stage 2.
> **Gdzie znaleźć:** gotowy plik **`output/training_curves.png`** albo `mainCnn.ipynb`, komórka 17.

---

## 6. Wyniki

### 6.1. Metoda klasyczna

Na zbiorze testowym (20% ze 114 obrazów, po odrzuceniu w treningu 10% najbardziej odstających próbek):

| Metryka | Wartość |
|---|---|
| **Dokładność (accuracy)** | **91,30%** |
| Komórki zdrowe | rozpoznane bezbłędnie |
| Błędy | 2 przypadki: komórki chore zaklasyfikowane jako zdrowe (fałszywie negatywne) |

Oba błędy dotyczyły komórek nowotworowych z **subtelnymi** zmianami (wysoka solidność ~0,97–0,99, niska heterogeniczność koloru), morfologicznie zbliżonych do zdrowych — czyli najtrudniejszych przypadków granicznych.

> 📷 **[ SCREEN 7 ]** — *Macierz pomyłek metody klasycznej*
> **Co pokazać:** macierz pomyłek (heatmapa 2×2) wraz z wypisaną skutecznością 91,30%.
> **Gdzie znaleźć:** `main.ipynb`, komórka 17 (wykres „Macierz pomyłek” + wydruk „SKUTECZNOŚĆ SYSTEMU”).

### 6.2. Metoda CNN — metryki niezależne od progu

Wartości liczone na całym zbiorze walidacyjnym (fold_2, 4 914 obrazów):

| Metryka | Wartość |
|---|---|
| **ROC AUC** | **0,7705** |
| **PR AUC** (klasa chora) | 0,7073 |
| Optymalny próg (macro-F1) | 0,56 (macro-F1 = 0,708) |

ROC AUC i PR AUC oceniają jakość rankingu prawdopodobieństw niezależnie od wybranego progu i są stabilniejszą miarą niż sama dokładność.

> 📷 **[ SCREEN 8 ]** — *Krzywe ROC i Precision-Recall*
> **Co pokazać:** dwie krzywe obok siebie — ROC (AUC = 0,770) i Precision-Recall (AUC = 0,707).
> **Gdzie znaleźć:** gotowy plik **`output/roc_pr_curves.png`** albo `mainCnn.ipynb`, komórka 18.

> 📷 **[ SCREEN 9 ]** — *Dobór progu decyzyjnego pod macro-F1*
> **Co pokazać:** wykres macro-F1 w funkcji progu, z zaznaczonym progiem optymalnym (0,56) i domyślnym (0,5).
> **Gdzie znaleźć:** gotowy plik **`output/threshold_tuning.png`** albo `mainCnn.ipynb`, komórka 20.

### 6.3. Metoda CNN — raporty klasyfikacji

**Próg domyślny 0,50:**

| Klasa | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Zdrowe (`hem`) | 0,820 | 0,545 | 0,655 | 2 457 |
| Białaczka (`all`) | 0,659 | 0,880 | 0,754 | 2 457 |
| **dokładność** | | | **0,713** | 4 914 |
| macro avg | 0,740 | 0,713 | 0,705 | 4 914 |

**Próg zoptymalizowany 0,56:**

| Klasa | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Zdrowe (`hem`) | 0,788 | 0,583 | 0,670 | 2 457 |
| Białaczka (`all`) | 0,669 | 0,843 | 0,746 | 2 457 |
| **dokładność** | | | **0,713** | 4 914 |
| macro avg | 0,728 | 0,713 | 0,708 | 4 914 |

Przy progu domyślnym sieć osiąga **recall klasy chorej = 0,88** — wykrywa 88% komórek nowotworowych, co jest pożądane z perspektywy medycznej. Obniżeniem ceny jest niższy recall klasy zdrowej (część zdrowych komórek jest oznaczana jako chore). Przesunięcie progu do 0,56 nieco wyrównuje obie klasy (recall zdrowych rośnie do 0,58 kosztem spadku recall chorych do 0,84), praktycznie nie zmieniając ogólnej dokładności.

> 📷 **[ SCREEN 10 ]** — *Macierze pomyłek CNN dla obu progów*
> **Co pokazać:** dwie macierze pomyłek obok siebie (próg 0,50 i próg 0,56). Opcjonalnie dołącz zrzut wydrukowanych raportów klasyfikacji.
> **Gdzie znaleźć:** gotowy plik **`output/confusion_matrices.png`** albo `mainCnn.ipynb`, komórka 22 (macierze + wydruk `classification_report`).

---

## 7. Analiza porównawcza

### 7.1. Zestawienie zbiorcze

| Aspekt | Metoda klasyczna | Metoda CNN (VGG16) |
|---|---|---|
| Reprezentacja cech | ręcznie projektowana | uczona automatycznie |
| Segmentacja komórki | wymagana | niepotrzebna |
| Normalizacja barwy | CLAHE (skala szarości) | Reinhard + CLAHE (kanał L) |
| Augmentacja danych | brak | offline (1:1) + warstwy Keras |
| Klasyfikator | ważony najbliższy centroid | VGG16 + głowa MLP (sigmoid) |
| Wielkość zbioru | 114 obrazów | ~9 700 obrazów (2 foldy) |
| Ewaluacja | podział 80/20 (ta sama pula) | międzyfoldowo (inne preparaty) |
| **Dokładność** | **91,3%** | **71,3%** |
| ROC AUC | nie raportowano | 0,77 |
| Interpretowalność | **wysoka** | niska (czarna skrzynka) |
| Koszt obliczeniowy | niski (sekundy, CPU) | wysoki (~godziny treningu, CPU) |
| Skalowalność | ograniczona (zależy od segmentacji) | **wysoka** |

### 7.2. Dlaczego CNN ma niższą dokładność? Porównanie nie jest „jeden do jednego”

Pozornie paradoksalny wynik (91,3% kontra 71,3%) **nie oznacza wyższości metody klasycznej**. Wynika on z fundamentalnych różnic w warunkach eksperymentu:

1. **Trudność i skala zbioru.** Metoda klasyczna testowana była na ok. 23 obrazach z małego, wyselekcjonowanego i czystego zbioru. CNN oceniono na **4 914** obrazach z pełnego, znacznie bardziej zaszumionego i zróżnicowanego zbioru C-NMC.
2. **Rygor ewaluacji.** Metoda klasyczna używa losowego podziału tej samej puli (trening i test pochodzą z tych samych preparatów). CNN walidowano **międzyfoldowo** — trening na foldzie 0, ocena na foldzie 2 (inne preparaty/pacjenci). To znacznie surowszy i bardziej realistyczny test generalizacji.
3. **Wariancja estymaty.** Dokładność liczona na 23 obrazach ma ogromny rozrzut (jeden błąd to ~4 punkty procentowe) i jest optymistycznie obciążona; metryka na ~5 tys. obrazów jest stabilna i wiarygodna.
4. **Zależność od segmentacji.** Skuteczność metody klasycznej stoi i upada na jakości maski. Na czystych, dobrze odseparowanych komórkach segmentacja działa znakomicie; na pełnym, trudnym zbiorze odsetek błędów segmentacji byłby znacznie wyższy, co zburzyłoby ekstrakcję cech.

Innymi słowy: metoda klasyczna działała w warunkach „łatwiejszych”, a sieć CNN — w „trudniejszych i uczciwszych”.

### 7.3. Charakter błędów z perspektywy medycznej

- **Metoda klasyczna:** wszystkie błędy to **fałszywie negatywne** (komórka chora uznana za zdrową) — przypadki o subtelnych zmianach. To najgroźniejszy diagnostycznie typ błędu, bo chory pacjent zostałby przeoczony.
- **Metoda CNN:** przy progu 0,50 osiąga **recall klasy chorej = 0,88**, a większość pomyłek to nadrozpoznawanie białaczki (zdrowe komórki oznaczane jako chore). To błąd mniej groźny klinicznie (fałszywy alarm kierowany do weryfikacji przez lekarza), choć obniża precyzję dla komórek zdrowych.

W zastosowaniu przesiewowym, gdzie celem jest „nie przeoczyć chorego”, profil błędów CNN jest korzystniejszy.

### 7.4. Ograniczenia porównania

- **Augmentacja zbioru walidacyjnego.** Wyrównanie klasy zdrowej zastosowano także w foldzie walidacyjnym, więc metryki CNN liczone są na sztucznym rozkładzie 50/50, a nie na naturalnym (~31% zdrowych). Dla w pełni rzetelnej oceny zaleca się ewaluację na **naturalnym, niezmienionym** rozkładzie oraz wykorzystanie zarezerwowanego foldu 1 jako niezależnego zbioru testowego.
- **Brak wspólnego benchmarku.** Obie metody nie zostały przetestowane na identycznym zbiorze testowym — to główne ograniczenie tego zestawienia. Pełni rzetelne porównanie wymagałoby uruchomienia metody klasycznej na foldach C-NMC.

---

## 8. Wnioski

1. **Metoda klasyczna** jest interpretowalna, szybka i bardzo skuteczna na małym, czystym zbiorze (91,3%), ale słabo się skaluje i jest krucha wobec błędów segmentacji oraz różnorodności danych rzeczywistych.
2. **Metoda CNN** nie wymaga segmentacji ani ręcznej inżynierii cech, naturalnie skaluje się do dużych zbiorów i została zweryfikowana w surowym scenariuszu międzyfoldowym (ROC AUC = 0,77), lecz na obecnym etapie osiąga niższą dokładność i pozostaje „czarną skrzynką” o wysokim koszcie obliczeniowym.
3. **Niższa dokładność CNN nie świadczy o gorszej metodzie** — wynika z trudniejszego zbioru i bardziej rygorystycznej ewaluacji. Metoda klasyczna działała w warunkach łatwiejszych.
4. **Rekomendacja:** kierunkiem rozwojowym jest podejście głębokie, ale wymaga ono dalszej pracy: połączenia foldów `fold_0 + fold_1` w większy zbiór treningowy, ewaluacji na naturalnym rozkładzie klas, eksperymentów z nowszymi architekturami (EfficientNet, ResNet), zastosowania funkcji straty typu *focal loss* oraz augmentacji testowej (TTA). Cechy ręcznie projektowane mogą posłużyć jako interpretowalny punkt odniesienia i element walidacji decyzji sieci.

---

## Bibliografia

- [Zbiór danych C-NMC Leukemia (Kaggle)](https://www.kaggle.com/datasets/avk256/cnmc-leukemia/data)
- [VGG16 — Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- E. Reinhard i in., *Color Transfer between Images*, IEEE Computer Graphics and Applications, 2001.
- [Dokumentacja TensorFlow / Keras](https://www.tensorflow.org/api_docs)
- [Dokumentacja scikit-learn](https://scikit-learn.org/stable/documentation.html)
- [Dokumentacja OpenCV](https://docs.opencv.org/) · [scikit-image](https://scikit-image.org/docs/stable/)

---

## Załącznik A — spis zrzutów ekranu do wstawienia

| # | Tytuł | Źródło |
|---|---|---|
| SCREEN 1 | Przykładowe obrazy wejściowe obu klas | `data/v1/healthy/`, `data/v1/cancer/` lub `data/fold_0/fold_0/hem/`, `.../all/` |
| SCREEN 2 | Rozróżnialność klas: kształt i tekstura | `main.ipynb`, komórka 13 |
| SCREEN 3 | Rozróżnialność klas: anomalie kolorystyczne | `main.ipynb`, komórki 14 i 15 |
| SCREEN 4 | Etapy przetwarzania obrazu (potok CNN) | `output/preprocessing_stages/_overview.png` lub `mainCnn.ipynb`, komórka 11 |
| SCREEN 5 | Architektura modelu (model.summary) | `mainCnn.ipynb`, komórka 13 |
| SCREEN 6 | Krzywe uczenia (accuracy / loss / AUC) | `output/training_curves.png` lub `mainCnn.ipynb`, komórka 17 |
| SCREEN 7 | Macierz pomyłek metody klasycznej | `main.ipynb`, komórka 17 |
| SCREEN 8 | Krzywe ROC i Precision-Recall | `output/roc_pr_curves.png` lub `mainCnn.ipynb`, komórka 18 |
| SCREEN 9 | Dobór progu decyzyjnego (macro-F1) | `output/threshold_tuning.png` lub `mainCnn.ipynb`, komórka 20 |
| SCREEN 10 | Macierze pomyłek CNN (oba progi) + raporty | `output/confusion_matrices.png` lub `mainCnn.ipynb`, komórka 22 |
