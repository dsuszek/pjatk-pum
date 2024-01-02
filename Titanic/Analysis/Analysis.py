import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import zscore

data = pd.read_csv('/Users/dominik/Library/Mobile Documents/com~apple~CloudDocs/PJATK/PUM/Titanic/titanic.csv')

# Eksploracyjna analiza danych
print(data.head().to_string())
print()
print()

stats = data.describe().to_string()
print(stats)

# Rekordy, dla których brakuje danych
missing_age = data[data['age'].isnull() == True]
# print(missing_age.to_string())

# Dla ilu rekordów brakuje danych w kolumnach: cabin, embarked, boat, body?
# Get the number of null records in each column
null_counts = data.isnull().sum()

# Display the number of null records
print('Null counts: ', null_counts)

data.drop(columns=['cabin', 'boat', 'body','home_dest'], inplace=True)
data['sex'] = data['sex'].str.capitalizea()

# correlation_matrix = data.iloc[:, 0:2].corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
# plt.show()

# print(data.to_string())


# Uzupełnianie danych w kolumnie age
# Wyodrębnij tytuły z kolumny 'Name'

data['title'] = data['name'].str.extract(r'(\bMiss\b|\bMaster\b|\bSir\b|\bDr\b|\bMrs\b|\bMr\b|\bMs\b)')

# Grupuj według tytułu i oblicz średnią wieku
average_age_by_title = data.groupby('title')['age'].mean()

# Uzupełnij brakujące wartości z kolumny age średnimi wartościami wyznaczonymi dla poszczególnuych grup
data['age'] = data.groupby('title')['age'].transform(lambda x: x.fillna(x.mean()))

# Usuń pozostałe rekordy, dla których age nadal jest równe null
data = data.dropna(subset=['age'])

# Wyświetl wyniki
print('average_age_by_title: ', average_age_by_title)

# Inżynieria zmiennych
# Dodaj zmienną family_size
data['family_size'] = data['parch'] + data['sibsp'] + 1

# Dodaj zmienną age_range
age_bins = [0, 6, 12, 18, float('inf')]
age_labels = ['Infant', 'Child', 'Teenager', 'Adult']
data['age_range'] = pd.cut(data['age'], bins=age_bins, labels=age_labels, right=False)

# Dodaj zmienną 'MPC'
data['MPC'] = data['age'] * data['pclass']

# Przygotuj wykres punktowy
data['z_score'] = zscore(data['age'])

# Zaznacz outliery na wykresie punktowym
plt.scatter(data.index, data['age'], label='Data Points')
plt.scatter(data[data['z_score'].abs() > 2.7].index, data[data['z_score'].abs() > 2.7]['age'], color='red', label='Outliers')

# Dodaj etykiety osi
plt.xlabel('Index')
plt.ylabel('Age')

# Dodaj tytuł wykresu
plt.title('Scatter Plot with Outliers')

# Dodaj legendę
plt.legend()

# Wyświetl wykres
plt.show()

# Zastąp wartości zakwalifikowane jako odstające średnią wartością wieku dla wszystkich pasażerów
# Oblicz średnią wartość wieku dla wszystkich pasażerów, wyłączająć wartości odstające
mean_age_without_outliers = data.loc[data['z_score'].abs() <= 2.7, 'age'].mean()
# Zastąp wartości odstające wartością średnią
data.loc[data['z_score'].abs() > 2.7, 'age'] = mean_age_without_outliers

print(data.sort_values('age').to_string())
