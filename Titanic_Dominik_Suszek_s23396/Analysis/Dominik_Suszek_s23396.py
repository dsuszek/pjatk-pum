import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

# Ładowanie danych
df = pd.read_csv('Analysis/titanic.csv')

######################################################################################################################
# 1. Analiza zbioru danych

df.info()
df.describe()
df.head()

# Macierz korelacji
numeric_data = df[df.select_dtypes('number').columns]
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title('Macierz korelacji')
plt.show()

# Dla ilu rekordów brakuje danych w kolumnach: age, cabin, embarked, boat, body?
null_counts = pd.DataFrame(df.isnull().sum(), columns=['Null Counts'])
plt.bar(null_counts.index, null_counts['Null Counts'], color='dimgray')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Columns')
plt.ylabel('Brakujące wartości')
plt.title('Brakujące wartości w zbiorze danych')
for i, value in enumerate(null_counts['Null Counts']):
    plt.text(i, value + 1, str(value), ha='center', va='bottom', fontsize=10)
plt.show()

# Wykres pokazujący zalezność pomiędzy klasą, którą podrózowała dana osoba, a tym, czy przezyła
plt.clf()
ax = sns.countplot(data=df, x='pclass', hue='survived')
ax.set_ylabel('Liczba osób')
ax.set_xlabel('Klasa biletu')
ax.tick_params(
    axis='x',                 
    top=False)     
# Add annotations on top of each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=8)
plt.show()

# Wykres pokazujący zaleznosc pomiędzy wiekiem, a liczbą osób, które przezyły
sns.distplot(df[df['age'].notnull() & (df['survived']==1)]['age'], 
             kde_kws={"label": "Przeżył"}, 
             bins=10)
sns.distplot(df[df['age'].notnull() & (df['survived']==0)]['age'], 
             kde_kws={"label": "Nie przeżył"}, 
             bins=10)
plt.show()


# Wykres pokazujący rozkłady empiryczne zmiennych sex, survived, pclass
plt.figure(figsize=(12, 6))

# Zmienna 'sex'
plt.subplot(1, 3, 1)
sns.countplot(x='sex', data=df)
plt.title('Histogram zmiennej \'sex\'')
plt.ylabel('Liczba rekordów')

# Zmienna 'survived'
plt.subplot(1, 3, 2)
sns.countplot(x='survived', data=df)
plt.title('Histogram zmiennej \'survived\'')
plt.ylabel('Liczba rekordów')

# Zmienna 'pclass'
plt.subplot(1, 3, 3)
sns.countplot(x='pclass', data=df)
plt.title('Histogram zmiennej \'pclass\'')
plt.ylabel('Liczba rekordów')

plt.tight_layout()
plt.show()

# Podział zbioru danych na dane treningowe i testowe - próbkowanie warstwowe na podstawie zmiennych 'survived', 'pclass', oraz 'sex'
stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_indices, test_indices in stratified_split.split(df, df[['survived', 'sex', 'pclass']]):
    train_dataset = df.loc[train_indices]
    test_dataset = df.loc[test_indices]

######################################################################################################################
# 2. Inżynieria cech
class FeaturesAdder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Dodaj zmienną family_size
        X['family_size'] = X['parch'] + X['sibsp'] + 1

        # Dodaj zmienną age_range
        age_bins = [0, 6, 12, 18, float('inf')]
        age_labels = ['1', '2', '3', '4']
        X['age_range'] = pd.cut(X['age'], bins=age_bins, labels=age_labels, right=False)

        # Dodaj zmienną 'MPC'
        X['MPC'] = X['age'] * X['pclass']

        return X

class AgeImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        # Uzupełnianie danych w kolumnie age
        # Wyodrębnione tytuły z kolumny 'Name'
        X['title'] = X['name'].str.extract(r'(\bMiss\b|\bMaster\b|\bSir\b|\bDr\b|\bMrs\b|\bMr\b|\bMs\b)')

        # Uzupełnij brakujące wartości z kolumny age średnimi wartościami wyznaczonymi dla poszczególnych grup
        X['age'] = X.groupby('title')['age'].transform(lambda x: x.fillna(x.mean()))

        # Usuń pozostałe rekordy, dla których age nadal jest równe null
        X = X.dropna(subset=['age'])

        return X

class OutliersRemover(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Zastąp wartości zakwalifikowane jako odstające średnią wartością wieku dla wszystkich pasażerów
        # Oblicz średnią wartość cenę biletu dla wszystkich pasażerów, wyłączając wartości odstające
        mean_fare_without_outliers = X.loc[X['fare'] <= 200, 'fare'].mean()
        # Zastąp wartości odstające wartością średnią
        X.loc[X['fare'] > 200, 'fare'] = mean_fare_without_outliers

        # Zastąp wartości zakwalifikowane jako odstające średnią wartością wieku dla wszystkich pasażerów
        # Oblicz średnią wartość wieku dla wszystkich pasażerów, wyłączając wartości odstające
        mean_age_without_outliers = X.loc[X['age'] <= 67, 'age'].mean()
        # Zastąp wartości odstające wartością średnią
        X.loc[X['age'] > 67, 'age'] = mean_age_without_outliers

        return X

class FareImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.dropna(subset=['fare'])

        return X


class FeaturesDropper(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(['name', 'cabin', 'boat', 'ticket', 'body', 'home_dest', 'embarked'], axis=1, errors="ignore")
    
class FeaturesAdjuster(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self
    
    def transform(self, X):
        # Przypisanie wartości 0 jeśli płeć to 'female', oraz wartości 1, jeśli płeć to 'male'
        sex_mapping = {'female': 0, 'male': 1}

        X['sex'] = X['sex'].map(sex_mapping)
        X['sex'] = X['sex'].astype(int)
        return X


# Dodaj potok transformujący
full_pipeline = Pipeline([
    ('age_imputer', AgeImputer()),
    ('fare_imputer', FareImputer()),
    ('features_adder', FeaturesAdder()),
    ('features_dropper', FeaturesDropper()),
    ('features_adjuster', FeaturesAdjuster()),
    ('outliers_remover', OutliersRemover()),
])

train_dataset_adjusted = full_pipeline.fit_transform(train_dataset)

# Przycięcie odstających wartości
# Zaznacz outliery dla zmiennej 'age' na wykresie punktowym
plt.clf()
plt.scatter(train_dataset.index, train_dataset['age'], label='Dane mieszczące się w normie')
plt.scatter(train_dataset[train_dataset['age'] > 67].index, train_dataset[train_dataset['age'] > 67]['age'], color='red', label='Wartości odstające')
# Dostosuj etykiety osi
plt.xticks([])
plt.ylabel('Wiek')
# Dodaj tytuł wykresu
plt.title('Wartości odstające dla zmiennej age')
# Dodaj legendę
plt.legend()
# Wyświetl wykres
plt.show()


# Zaznacz outliery dla zmiennej 'fare' na wykresie punktowym
plt.clf()
plt.scatter(train_dataset.index, train_dataset['fare'], label='Dane mieszczące się w normie')
plt.scatter(train_dataset[train_dataset['fare'] > 200].index, train_dataset[train_dataset['fare'] > 200]['fare'], color='red', label='Wartości odstające')
# Dostosuj etykiety osi
plt.xticks([])
plt.ylabel('Opłata za bilet')
# Dodaj tytuł wykresu
plt.title('Wartości odstające dla zmiennej fare')
# Dodaj legendę
plt.legend()
# Wyświetl wykres
plt.show()


X_train_adjusted = train_dataset_adjusted.drop(['title', 'survived'], axis=1)
y_train = train_dataset_adjusted['survived']

# Normalizacja danych numerycznych
scaler = MinMaxScaler()
X_train_adjusted_scaled = scaler.fit_transform(X_train_adjusted)

# Trenowanie modelu
rfc = RandomForestClassifier()

param_grid = [
    {"n_estimators": [10, 100, 200, 500], "max_depth": [None, 5, 10], "min_samples_split": [2, 3, 4]}
]

grid_search = GridSearchCV(rfc, param_grid, cv=3, scoring="accuracy", return_train_score=True)
grid_search.fit(X_train_adjusted_scaled, y_train)

grid_search.best_params_
final_rfc = grid_search.best_estimator_
final_rfc_results = grid_search.cv_results_
for mean_score, params in zip(final_rfc_results["mean_test_score"], final_rfc_results["params"]):
    print(np.sqrt(mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

test_dataset_adjusted = full_pipeline.fit_transform(test_dataset)
X_test_prepared = test_dataset_adjusted.drop(['title', 'survived'], axis=1)
y_test = test_dataset_adjusted['survived']
X_test_prepared = scaler.fit_transform(X_test_prepared)

# Ewaluacja modeli, dostrajanie, analiza
y_pred = final_rfc.predict(X_test_prepared)

accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
cm