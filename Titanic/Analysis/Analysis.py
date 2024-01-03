import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as mcs
import missingno as msno
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

# Ładowanie danych
df = pd.read_csv('/Users/dominiksuszek/git/pjatk-pum/Titanic/titanic.csv')

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
null_counts = pd.DataFrame(df.isnull().sum())
print(null_counts)

msno.bar(null_counts, figsize=(10, 6), fontsize=12, labels=None, label_rotation=45, log=False, color='dimgray', filter=None, n=0, p=0, sort=None, ax=None, orientation=None)
plt.show()

# Wykres pokazujący zalezność pomiędzy klasą, którą podrózowała dana osoba, a tym, czy przezyła
plt.clf()
sns.catplot(data=df,
            x='pclass',
            kind='count',
            hue='survived')
plt.show

# Wykres pokazujący zaleznosc pomiędzy wiekiem, a liczbą osób, które przezyły
sns.distplot(df[df['age'].notnull() & (df['survived']==1)]['age'], 
             kde_kws={"label": "Survived"}, 
             bins=10)
sns.distplot(df[df['age'].notnull() & (df['survived']==0)]['age'], 
             kde_kws={"label": "Not Survived"}, 
             bins=10)
plt.show()

# Podział zbioru danych na dane treningowe i testowe
stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_indices, test_indices in stratified_split.split(df, df[['survived', 'pclass', 'sex']]):
    train_dataset = df.loc[train_indices]
    test_dataset = df.loc[test_indices]

# Sprawdzenie, czy oba zbiory danych - treningowy i testowy zawierają takie same proporcje osób, które przezyly
plt.clf()
plt.subplot(1, 2, 1)
train_dataset['survived'].hist()
plt.tick_params(
    axis='x',          
    which='both',      
    bottom=False,      
    top=False,         
    labelbottom=False) 
plt.title('Train dataset')

plt.subplot(1, 2, 2)
test_dataset['survived'].hist()
plt.tick_params(
    axis='x',          
    which='both',      
    bottom=False,      
    top=False,         
    labelbottom=False) 
plt.title('Test dataset')
plt.show()

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

        return 

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
        return X.drop(columns=['name', 'cabin', 'boat', 'ticket', 'body', 'home_dest', 'embarked'], axis=1, errors="ignore")

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
trans_pipeline = Pipeline([
    # ('age_imputer', AgeImputer()),
    ('fare_imputer', FareImputer()),
    ('features_adder', FeaturesAdder()),
    # ('features_dropper', FeaturesDropper()),
    # ('features_adjuster', FeaturesAdjuster()),
    ('outliers_remover', OutliersRemover()),
])
train_dataset_adjusted = FareImputer.fit_transform(train_dataset)
train_dataset_adjusted = trans_pipeline.fit_transform(train_dataset)

# Przycięcie odstających wartości
# Zaznacz outliery dla zmiennej 'age' na wykresie punktowym
plt.clf()
plt.scatter(train_dataset.index, train_dataset['age'], label='Data Points')
plt.scatter(train_dataset[train_dataset['age'] > 67].index, train_dataset[train_dataset['age'] > 67]['age'], color='red', label='Outliers')
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
plt.clf()
plt.scatter(train_dataset.index, train_dataset['fare'], label='Data Points')
plt.scatter(train_dataset[train_dataset['fare'] > 200].index, train_dataset[train_dataset['fare'] > 200]['fare'], color='red', label='Outliers')
# Dodaj etykiety osi
plt.xlabel('Index')
plt.ylabel('Fare')
# Dodaj tytuł wykresu
plt.title('Scatter Plot with Outliers')
# Dodaj legendę
plt.legend()
# Wyświetl wykres
plt.show()


X_train_adjusted = train_dataset_adjusted.drop('survived', axis=1)
y_train_adjusted = train_dataset_adjusted['survived']

print(X_train_adjusted.dtypes)
print(X_train_adjusted.head())

# Normalizacja danych numerycznych
scaler = MinMaxScaler()
X_train_adjusted = scaler.fit_transform(X_train_adjusted)
print(X_train_adjusted_scaled.head())

# Trenowanie modelu
tree_classifier = DecisionTreeClassifier(random_state=13)
tree_classifier.fit(X_train_adjusted, y_train)
y_pred = tree_classifier.predict(X_test)

print(mcs.accuracy_score(y_test, y_pred))

# Ewaluacja modeli, dostrajanie, analiza