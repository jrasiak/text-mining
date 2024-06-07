# text-mining project by Jan Rasiak and Adrian Wdowczyk

Raport z analizy danych i klasyfikacji wiadomości jako spam lub ham
1. Wprowadzenie
Celem tego projektu jest stworzenie klasyfikatora, który będzie w stanie odróżniać wiadomości spam od wiadomości ham (nie-spam). W projekcie wykorzystano zbiór danych zawierający wiadomości oznaczone jako spam lub ham oraz przeprowadzono szereg operacji przetwarzania danych, wektoryzacji tekstu, próbkowania, trenowania modelu oraz analizy wyników.

2. Przetwarzanie danych
a) Wczytanie danych
Dane zostały wczytane z pliku CSV przy użyciu biblioteki Pandas:

```self.data = pd.read_csv(data_path, encoding='latin1')```

Uzasadnienie: Pandas to potężne narzędzie do manipulacji danymi, które pozwala na łatwe wczytywanie, przekształcanie i analizowanie dużych zbiorów danych.

b) Usunięcie niepotrzebnych kolumn
Z danych usunięto kolumny, które nie mają znaczenia dla analizy:
 
```self.data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)```

Uzasadnienie: Usunięcie niepotrzebnych kolumn redukuje rozmiar danych i upraszcza analizę, co jest istotne dla utrzymania przejrzystości i efektywności dalszych operacji.

c) Zmiana nazw kolumn
Kolumny zostały odpowiednio nazwane, aby były bardziej czytelne:

```self.data.rename(columns={'v1': 'target', 'v2': 'Message'}, inplace=True)```

Uzasadnienie: Przejrzyste nazwy kolumn ułatwiają zrozumienie danych i dalsze przetwarzanie.

d) Kodowanie etykiet
Kolumna target została zakodowana przy użyciu LabelEncoder, co zamienia wartości tekstowe na numeryczne:

```self.data['target'] = LabelEncoder().fit_transform(self.data['target'])```

Uzasadnienie: Modele uczenia maszynowego wymagają danych numerycznych do przetwarzania. LabelEncoder jest łatwym i efektywnym sposobem na konwersję kategorii do formy numerycznej.

e) Normalizacja i lematyzacja tekstu
Tekst został znormalizowany (usunięcie znaków specjalnych, konwersja na małe litery) oraz przeprowadzono lematyzację (zamiana słów na ich podstawowe formy):

```self.data['Message'] = self.data['Message'].apply(self.processor.normalize).apply(self.processor.lemmatize)```

Uzasadnienie: Normalizacja i lematyzacja redukują szum w danych tekstowych, co pozwala na bardziej precyzyjną analizę i modelowanie. Usunięcie znaków specjalnych i konwersja na małe litery ujednolicają dane, a lematyzacja pomaga w zredukowaniu liczby różnych form tego samego słowa.

f)Usunięto duplikaty z danych:

```self.data = self.data.drop_duplicates(keep='first')```

Uzasadnienie: Usunięcie duplikatów poprawia jakość danych i zapewnia, że model nie będzie uczył się na tych samych przykładach wielokrotnie, co mogłoby prowadzić do przetrenowania.

3. Wektoryzacja tekstu
Zamiana tekstu na reprezentację liczbową przy użyciu TfidfVectorizer:

```
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X = vectorizer.fit_transform(self.data['Message'])
```

Uzasadnienie: TF-IDF (Term Frequency-Inverse Document Frequency) jest skuteczną metodą wektoryzacji tekstu, która uwzględnia zarówno częstość występowania słów, jak i ich znaczenie w kontekście całego zbioru dokumentów. Pomaga to w identyfikacji istotnych słów dla klasyfikacji.

5. Próbkowanie danych
Aby zbalansować klasy w danych, zastosowano techniki SMOTE (oversampling) oraz RandomUnderSampler (undersampling):

```
over = SMOTE(sampling_strategy=1)
under = RandomUnderSampler(sampling_strategy=0.4)
pipeline = Pipeline(steps=[('under', under), ('over', over)])
X_resampled, y_resampled = pipeline.fit_resample(X, y)
```

Uzasadnienie: W przypadku niezbalansowanych danych klasyfikacyjnych, modele mogą być stronnicze w kierunku dominującej klasy. Zastosowanie technik oversamplingu (SMOTE) i undersamplingu (RandomUnderSampler) pomaga w zbalansowaniu klas, co prowadzi do bardziej rzetelnych i dokładnych wyników.

7. Trenowanie modelu
Model RandomForestClassifier został wytrenowany na przetworzonych danych:

```self.model.fit(X_train, y_train)```

Uzasadnienie: RandomForestClassifier jest wszechstronnym i skutecznym modelem klasyfikacyjnym, który dobrze radzi sobie z różnymi typami danych i potrafi uchwycić złożone zależności w danych. Jego wbudowane mechanizmy radzenia sobie z nadmiernym dopasowaniem (overfitting) sprawiają, że jest to dobry wybór dla tego zadania.

9. Ewaluacja modelu
Model został oceniony przy użyciu metryk takich jak raport klasyfikacji, macierz konfuzji oraz ROC AUC Score:

```
predictions = self.model.predict(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print("ROC AUC Score:", roc_auc_score(y_test, predictions))
```

Uzasadnienie: Użycie różnych metryk oceny pozwala na wszechstronną analizę wydajności modelu. Raport klasyfikacji dostarcza szczegółowych informacji na temat precyzji, recall i F1-score dla każdej klasy, macierz konfuzji pozwala zrozumieć rozkład błędów, a ROC AUC Score dostarcza ogólnego wskaźnika skuteczności modelu.

11. Wizualizacja danych
Przeprowadzono wizualizację rozkładu klas w danych:

```
l = self.data['target'].value_counts()
colors = ['#8BC34A','#B2EBF2']
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
ax[0].pie(l, explode=[0,0.1], autopct='%1.1f%%', shadow=True, labels=['Ham', 'Spam'], colors=colors)
ax[0].set_title('Target (%)')
sns.countplot(x='target', data=self.data, palette=colors, edgecolor='black', ax=ax[1])
ax[1].set_title('Number of Target')
plt.show()
```

Uzasadnienie: Wizualizacja danych pomaga w zrozumieniu rozkładu klas oraz identyfikacji potencjalnych problemów z niezbalansowanymi danymi. Graficzna reprezentacja danych jest często łatwiejsza do zinterpretowania niż same liczby.

8. Analiza wyników
Model RandomForestClassifier uzyskał następujące wyniki:

Raport klasyfikacji: Pokazuje metryki takie jak precyzja, recall, F1-score dla każdej klasy.
Macierz konfuzji: Pokazuje liczbę poprawnie i niepoprawnie zaklasyfikowanych przykładów dla każdej klasy.
ROC AUC Score: Pokazuje skuteczność klasyfikatora na podstawie krzywej ROC.
Przykładowe wyniki mogą wyglądać następująco (dane wyjściowe zależą od konkretnego uruchomienia):

```
              precision    recall  f1-score   support
           0       0.98      0.99      0.99       145
           1       0.97      0.94      0.96        50
    accuracy                           0.98       195
    
   macro avg       0.97      0.96      0.97       195
   
weighted avg       0.98      0.98      0.98       195

ROC AUC Score: 0.965
```

ANALIZA WYNIKÓW: 

Precyzja: Wartości 0.98 dla klasy 0 i 0.97 dla klasy 1 oznaczają, że model rzadko popełnia błędy w przewidywaniu obu klas.

Czułość: Wartości 0.99 dla klasy 0 i 0.94 dla klasy 1 pokazują, że model jest bardzo skuteczny w identyfikacji klasy 0, nieco mniej skuteczny w przypadku klasy 1.

F1-Score: Wyniki 0.99 dla klasy 0 i 0.96 dla klasy 1 wskazują na bardzo dobrą wydajność modelu w obu klasach, przy czym klasa 0 jest nieco lepiej obsługiwana.

Dokładność: Dokładność 0.98 oznacza, że 98% wszystkich przewidywań modelu są poprawne. Jest to ogólny wskaźnik wydajności modelu, który uwzględnia zarówno pozytywne, jak i negatywne wyniki.

Macro AVG: Wartości 0.97 dla precyzji, 0.96 dla czułości i 0.97 dla F1-score pokazują, że model ogólnie dobrze sobie radzi w obu klasach, ale klasa 1 jest nieco mniej dokładnie przewidywana niż klasa 0.

Weighted AVG: Wartości 0.98 dla precyzji, 0.98 dla czułości i 0.98 dla F1-score pokazują, że model jest bardzo wydajny ogólnie, z lekkim przeważeniem na korzyść klasy 0 ze względu na większą liczbę przypadków w tej klasie.

ROC AUC Score: Wynik 0.965 wskazuje, że model ma bardzo dobrą zdolność do rozróżniania między klasami.

Uzasadnienie: Wyniki wskazują na wysoką skuteczność modelu w klasyfikacji wiadomości jako spam lub ham. Wysokie wartości precyzji, recall oraz F1-score świadczą o tym, że model dobrze radzi sobie z obydwoma klasami. Wysoki ROC AUC Score potwierdza, że model ma dobrą zdolność

