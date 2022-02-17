


# ALL THE SAME IDENTICAL CODE IS ON THE GOOGLE COLAB, ON GITHUB
# This file was created as a compact version of the notebook.


import numpy as np
import pandas as pd

url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'
owid = pd.read_csv(url)

cols_to_remove = owid[[column for column in owid.columns if owid[column].count() < 100000]]

owid.drop(cols_to_remove, axis=1, inplace=True)

owid.drop('iso_code', axis=1, inplace=True)

owid['continent'].fillna(value='World', inplace=True)

cols_to_remove = [
    'new_cases',
    'new_deaths',
    'new_cases_per_million',
    'new_deaths_per_million'
    ]

owid.drop(cols_to_remove, axis=1, inplace=True)

index_stringency_index = owid.columns.get_loc('stringency_index')

counter = 0

while counter < index_stringency_index:
  owid.iloc[:, counter].fillna(0, inplace=True)
  counter += 1


owid.loc[owid['location'] == 'International', 'population'] = 7900000000

owid.loc[owid['location'] == 'Northern Cyprus', 'population'] = 326000


owid.drop('population_density', axis=1, inplace=True)

continents = owid['continent'].unique()
continents.sort()

def filling_the_na_values_with_means(column):
  list_of_means = owid.groupby('continent')[column].mean().tolist()
  zipped_list = [(x, y) for x, y in zip(continents, list_of_means)]

  for i in zipped_list:
    for index in owid.index:
      if owid.loc[index, 'continent'] == i[0] and np.isnan(owid.loc[index, column]):
        owid.loc[index, column] = i[1]

check_list_of_means = owid.groupby('continent')['gdp_per_capita'].mean().tolist()
check_zipped_list = [(x, y) for x, y in zip(continents, list_of_means)]
check_zipped_list

for column in owid.columns[index_stringency_index:]:
  filling_the_na_values_with_means(column)

owid['smokers_pop'] = owid['female_smokers'] + owid['male_smokers']

owid.drop(['female_smokers', 'male_smokers'], axis=1, inplace=True)

owid.drop(labels=[x for x in owid.columns if 'new' in x], axis=1, inplace=True)

owid = owid.groupby('location')[owid.columns].max()

owid.reset_index(drop=True, inplace=True)

owid.drop(labels='date', axis=1, inplace=True)

owid.sort_values(['continent', 'location'], ignore_index=True, inplace=True)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
plt.xlabel('Total Deaths')
plt.ylabel('Total Cases')
plt.scatter(
    owid.groupby('location')['total_cases'].max(),
    owid.groupby('location')['total_deaths'].max(),
    c=[range(owid['location'].count())]
    )

plt.figure(figsize=(10, 6))
plt.xlabel('Total deaths per million')
plt.ylabel('Total cases per million')
plt.scatter(
    owid.groupby('location')['total_cases_per_million'].max(),
    owid.groupby('location')['total_deaths_per_million'].max(),
    c=[range(owid['location'].count())]
    )

owid.nlargest(1, 'total_cases') == owid.nlargest(1, 'total_deaths')

owid.iloc[214]

macro_owid = owid[owid['continent'] == 'World']

owid.drop(owid[owid['continent'] == 'World'].index, inplace=True)

sns.set_style('darkgrid')
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', fontsize=13)
plt.rc('font', size=13)

plt.figure(figsize=(16, 11))
sns.heatmap(owid.corr(), annot=True, cmap='viridis')

possible_insights = owid[['continent', 'total_cases_per_million', 'total_deaths_per_million', 'median_age', 'reproduction_rate', 'stringency_index', 'life_expectancy', 'human_development_index']]
plt.figure(figsize=(10, 8), tight_layout=True)
sns.heatmap(possible_insights.corr(), annot=True, cmap='viridis')

plt.figure(figsize=(23, 8))
plt.subplot(1, 2, 1)
sns.violinplot(data=possible_insights, y='total_deaths_per_million', x='continent')
plt.xlabel('')
plt.ylabel('Total deaths per million')
plt.subplot(1, 2, 2)
sns.violinplot(data=possible_insights, y='median_age', x='continent')
plt.xlabel('')
plt.ylabel('Median Age')

plt.figure(figsize=(23, 8))
plt.subplot(1, 2, 1)
sns.boxplot(data=possible_insights, y='total_cases_per_million', x='continent')
plt.xlabel('')
plt.ylabel('Total cases per million')
plt.subplot(1, 2, 2)
sns.boxplot(data=possible_insights, y='human_development_index', x='continent')
plt.xlabel('')
plt.ylabel('Human development index')

from sklearn.linear_model import LinearRegression

total_cases_per_million = np.array(owid['total_cases_per_million'])
total_cases_per_million = total_cases_per_million.reshape(-1, 1)

total_deaths_per_million = np.array(owid['total_deaths_per_million'])
total_deaths_per_million = total_deaths_per_million.reshape(-1, 1)

line_fitter = LinearRegression()
line_fitter.fit(total_cases_per_million, total_deaths_per_million)
total_deaths_per_million_predict = line_fitter.predict(total_cases_per_million)

plt.figure(figsize=(10, 8))
plt.title('Prediction of number of deaths based on number of cases')
plt.plot(total_cases_per_million, total_deaths_per_million_predict)
plt.plot(total_cases_per_million, total_deaths_per_million, 'o')
plt.xlabel('Cases per million')
plt.ylabel('Deaths per million')
plt.show()

from sklearn.model_selection import train_test_split

x = owid[['median_age', 'reproduction_rate', 'stringency_index', 'life_expectancy', 'human_development_index']]
y = owid['total_deaths_per_million']
z = owid['total_cases_per_million']

# First test
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

# Second test
x_2_train, x_2_test, z_train, z_test = train_test_split(x, z, train_size=0.8, test_size=0.2)

mlr = LinearRegression()
mlr_2 = LinearRegression()

# First fit
mlr_model = mlr.fit(x_train, y_train)

# Second fit
mlr_model_2 = mlr_2.fit(x_2_train, z_train)

y_predict = mlr_model.predict(x_test)
z_predict = mlr_model_2.predict(x_2_test)

plt.figure(figsize = (20, 8))
plt.subplot(1, 2, 1)
plt.xlabel('Deaths per million')
plt.ylabel('Predicted Deaths')
plt.plot(y_test, y_test)
plt.plot(y_test, y_predict, 'o')
plt.subplot(1, 2, 2)
plt.xlabel('Cases per million')
plt.ylabel('Predicted cases')
plt.plot(z_test, z_test)
plt.plot(z_test, z_predict, 'o')

print('Train 1 score:')
print(mlr_model.score(x_train, y_train))
print('\n')
print('Train 2 score:')
print(mlr_model_2.score(x_2_train, z_train))

print('Test 1 score:')
print(mlr_model.score(x_test, y_test))
print('\n')
print('Train 2 score:')
print(mlr_model_2.score(x_2_test, z_test))

mlr_model.coef_
mlr_model_2.coef_

# First group of independent variables
# w - y
w = x.drop(['stringency_index', 'life_expectancy'], axis=1)
w_train, w_test, y_train, y_test = train_test_split(w, y, train_size = 0.8, test_size = 0.2, random_state=6)
lm = LinearRegression()
model = lm.fit(w_train, y_train)
y_predict= lm.predict(w_test)

plt.figure(figsize = (13, 8))
plt.xlabel('Deaths per million')
plt.ylabel('Predicted Deaths')
plt.plot(y_test, y_test)
plt.plot(y_test, y_predict, 'o')

print("Train score:")
print(w_model.score(w_train, y_train))

print("Test score:")
print(w_model.score(w_test, y_test))
