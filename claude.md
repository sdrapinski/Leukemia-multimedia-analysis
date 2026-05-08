### o projekcie

jest to projekt na studia wykrywania komorek rakowych białaczki na podstawie zdjec.
w pliku main.ipynb jest poprzednia wersja projektu ktora miała działać bez uzycia ai z zeszłego semestru.
aktualnie mamy wykorzystać sieci neuronowe do tego z uzyciem konwolusyjnych sieci neuronowych CNN. 

### technologia
Python 3.13.5

### foldery
/data folder z danymi wejsciowymi
/output folder na dane wyjsciowe
/src klasy pythonowe wykorzystywane w main
/utils dodatkowe utilsy uzywane przez projekt

/data/fold_x/foldx folder /all zawiera komorki nowotworowe a /hem zawiera komorki zdrowe  

### Plan:
Przeanalizuj mainCnn oraz otrzymane wyniki, dodaj sekcje z wagami dla modelu, 
 precision    recall  f1-score   support

   Zdrowe (hem)       0.55      0.59      0.57      1096
Białaczka (all)       0.81      0.79      0.80      2457

aktualnie precyzja nie jest za duza, szczegolnie przy zdrowych głownie przez to ze dane wejsciowe to w wiekszosci chore komorki.
Popraw działanie modelu w celu poprawy jakosci i odkrywania.