import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as mcs
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

# Ładowanie danych
data = pd.read_csv('/Users/dominiksuszek/git/pjatk-pum/Titanic/titanic.csv')

######################################################################################################################
# 1. Analiza zbioru danych
print(data.head().to_string())
print(data.describe().to_string())

# Macierz korelacji
data.describe()
numeric_data = data[['pclass', 'age', 'sibsp', 'parch', 'fare', 'body']]
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.show()

# Dla ilu rekordów brakuje danych w kolumnach: age, cabin, embarked, boat, body?
null_counts = data.isnull().sum()

# Wyświetl informację o brakujacych rekordach
print('Null counts: ', null_counts)

# Podział zbioru danych na dane treningowe i testowe
stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_indices, test_indices in stratified_split.split(data, data[['survived', 'pclass', 'sex']]):
    train_dataset = data.loc[train_indices]
    test_dataset = data.loc[test_indices]

# Sprawdzenie, czy oba zbiory danych - treningowy i testowy zawierają takie same proporcje osób, które przezyly
plt.subplot(1, 2, 1)
test_dataset['survived'].hist()

plt.subplot(1, 2, 2)
test_dataset['survived'].hist()
plt.show()

X_train = train_dataset.drop('survived', axis=1)
y_train = train_dataset['survived']
X_test = test_dataset.drop('survived', axis=1)
y_test = test_dataset['survived']
print(X_train.dtypes)
######################################################################################################################
# 2. Inżynieria cech

# Przygotuj wykres punktowy dla zmiennej 'age'
X_train['z_score'] = zscore(X_train['age'])

# Zaznacz outliery dla zmiennej 'age' na wykresie punktowym
plt.scatter(X_train.index, X_train['age'], label='Data Points')
plt.scatter(X_train[X_train['z_score'].abs() > 2.5].index, X_train[X_train['z_score'].abs() > 2.5]['age'], color='red', label='Outliers')

# Dodaj etykiety osi
plt.xlabel('Index')
plt.ylabel('Age')

# Dodaj tytuł wykresu
plt.title('Scatter Plot with Outliers')

# Dodaj legendę
plt.legend()

# Wyświetl wykres
plt.show()

# Zaznacz outliery dla zmiennej 'fare' na wykresie punktowym
plt.scatter(X_train.index, X_train['fare'], label='Data Points')
plt.scatter(X_train[X_train['z_score'].abs() > 2.7].index, X_train[X_train['z_score'].abs() > 2.7]['fare'], color='red', label='Outliers')

# Dodaj etykiety osi
plt.xlabel('Index')
plt.ylabel('Fare')

# Dodaj tytuł wykresu
plt.title('Scatter Plot with Outliers')

# Dodaj legendę
plt.legend()


class FeaturesAdder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Dodaj zmienną family_size
        X['family_size'] = X['parch'] + X['sibsp'] + 1

        # Dodaj zmienną age_range
        age_bins = [0, 6, 12, 18, float('inf')]
        age_labels = ['Infant', 'Child', 'Teenager', 'Adult']
        X['age_range'] = pd.cut(X['age'], bins=age_bins, labels=age_labels, right=False)

        # Dodaj zmienną 'MPC'
        X['MPC'] = X['age'] * X['pclass']

        return X

class FeaturesDropper(BaseEstimator, TransformerMixin):
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        return X.drop(columns=['name', 'cabin', 'boat', 'ticket', 'body','home_dest', 'embarked', 'age_range'], axis=1)

class FeaturesAdjuster(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self
    
    def transform(self, X):
        # Przypisanie wartości 0 jeśli płeć to 'female', oraz wartości 1, jeśli płeć to 'male'
        sex_mapping = {'female': 0, 'male': 1}

        X['sex'] = X['sex'].map(sex_mapping)
        X['sex'] = X['sex'].astype(int)
        return X
    
class Scaler(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self
    
    def transform(self, X):
        scaler = MinMaxScaler()
        X['pclass'] = scaler.fit_transform(X['pclass'])
        X['age'] = scaler.fit_transform(X['age'])
        X['sibsp'] = scaler.fit_transform(X['sibsp'])
        X['parch'] = scaler.fit_transform(X['parch'])
        X['fare'] = scaler.fit_transform(X['fare'])
        X['body'] = scaler.fit_transform(X['body'])

# Dodaj potok transformujący
trans_pipeline = Pipeline([
    ('features_adder', FeaturesAdder()),
    ('features_dropper', FeaturesDropper()),
    ('features_adjuster', FeaturesAdjuster()),
])

X_train_adjusted = trans_pipeline.transform(X_train)
print(X_train_adjusted.head())

##############################################################################
# 4. Uzupełnienie brakujących danych


# Uzupełnianie danych w kolumnie age
# Wyodrębnione tytuły z kolumny 'Name'

X_train['title'] = X_train['name'].str.extract(r'(\bMiss\b|\bMaster\b|\bSir\b|\bDr\b|\bMrs\b|\bMr\b|\bMs\b)')

# Grupuj według tytułu i oblicz średnią wieku
average_age_by_title = X_train.groupby('title')['age'].mean()

# Uzupełnij brakujące wartości z kolumny age średnimi wartościami wyznaczonymi dla poszczególnuych grup
X_train['age'] = X_train.groupby('title')['age'].transform(lambda x: x.fillna(x.mean()))

# Usuń pozostałe rekordy, dla których age nadal jest równe null
X_train = X_train.dropna(subset=['age'])

# Wyświetl wyniki
print('average_age_by_title: ', average_age_by_title)


# Zastąp wartości zakwalifikowane jako odstające średnią wartością wieku dla wszystkich pasażerów
# Oblicz średnią wartość wieku dla wszystkich pasażerów, wyłączająć wartości odstające
mean_age_without_outliers = X_train.loc[X_train['z_score'].abs() <= 2.7, 'age'].mean()
# Zastąp wartości odstające wartością średnią
X_train.loc[X_train['z_score'].abs() > 2.7, 'age'] = mean_age_without_outliers

# Nadal mamy jeden brakujący rekord dla zmiennej 'fare', oraz dwa brakujące rekordy dla zmiennej 'embarked'
X_train = X_train.dropna(subset=['fare', 'embarked'])

# Wyznacz liczbę pustych rekordów w kazdej z kolumn
null_counts = X_train.isnull().sum()

# Wyświetl liczbę pustych rekordów
print('Null counts: ', null_counts)

# Wyświetl wykres
# plt.show()

# print(scaler.fit(X_train['pclass', 'age', 'sibsp', 'parch', 'fare', 'family_size', 'MPC']))
# print(X_train.describe())


tree_classifier = DecisionTreeClassifier(random_state=13)
tree_classifier.fit(X_train_adjusted, y_train)
y_pred = tree_classifier.predict(X_test)

print(mcs.accuracy_score(y_test, y_pred))