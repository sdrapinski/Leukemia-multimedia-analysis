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
0: Przeanalizuj projekt oraz foldery.
1. stare dane wejsciowe zostały przeniesione do /data/v1 zaktualizuj scieżki do folderów w plikach które z nich korzystały
2. Popraw plik mainCnn pod kątem funkcjonalnosci, jakości kodu oraz by używał nowych danych a nie starych(pewnie bedziesz musiał poprawic sciezki do plikow takze), 
niech uzyje /fold_0 lub fold_1 jako zbioru treningowego a fold 2 jako zbioru walidacyjnego