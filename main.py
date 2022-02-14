import numpy as np
import pandas as pd

# reading instead of getting the dataset because it's updated daily

url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'
owid = pd.read_csv(url)

# owid stands for Our World In Data
# securing the dataset with a copy

owid_backup = owid.copy()

# taking a peek at the data

print(owid.head(10))
print(owid.shape)
print(owid.describe())
print(owid.info())

# before we do some data exploration we absolutely have to clean up the dataset,
# as it misses quite many values.
# first let's drop some columns: it's reasonable to say that
# columns < 100 000 values cannot be representative (by 12/02/2021 there are
# more than 160 000 rows)
# first of all let's find the columns.count() < 100 000

cols_to_remove = owid[[column for column in owid.columns if owid[column].count() < 100000]]

# dropping the columns

owid.drop(cols_to_remove, axis=1, inplace=True)
print(owid.info())

# much better
# the iso_code it's not that important

owid.drop('iso_code', axis=1, inplace=True)
print(owid.info())

# strange to read that continent misses some values

print(owid[owid['continent'].isnull()])
print(owid[owid['continent'].isnull()]['location'].unique())

# we can see that 'location' for NaN 'continents' are either continents
# themselves, income classification and World
# we can say that their 'continent' can be 'World'

owid['continent'].fillna(value='World', inplace=True)
print(str(owid['continent'].count()) + ' true values out of ' + str(len(owid)))

# from both the info() and github repo we see that the features expressing
# cases, vaccinations, deaths... are sided by their 'smoothed' feature
# the 'smoothed' data refers to the correction of the data by means of
# some probabilistic models, especially useful when the data come from
# third-world countries and war-zones
# we can say that 'smoothed' features are more accurate than their counterparts

cols_to_remove = [
    'new_cases',
    'new_deaths',
    'new_cases_per_million',
    'new_deaths_per_million'
    ]

owid.drop(cols_to_remove, axis=1, inplace=True)
print(owid.info())

# now it's time to replace the NaN values

print(owid[owid['total_cases'].isnull()]['location'].unique())
print(owid[owid['location'] == 'Albania'].head())
print(owid[owid['location'] == 'Taiwan'].head())
print(owid[owid['location'] == 'Taiwan'].tail())

# we note that as far as the values of cases and deaths are concerned,
# as for example for Taiwan, NaN values come from the first few rows.
# these first rows refer to the beginning of the pandemic, so it is reasonable
# to say that these dates hold 0 new_cases and 0 new_deaths, beacause were
# not even detected by local governments
# let's fill these rows with 0s

index_stringency_index = owid.columns.get_loc('stringency_index')
counter = 0

while counter < index_stringency_index:
  owid.iloc[:, counter].fillna(0, inplace=True)
  counter += 1

print(owid.info())

# let's fix the population

print(owid[owid['population'].isnull()]['location'].unique())
owid.loc[owid['location'] == 'International', 'population'] = 7900000000
owid.loc[owid['location'] == 'Northern Cyprus', 'population'] = 326000
str(owid['population'].count()) + ' truthy values out of ' + str(len(owid))
print(owid[owid['population_density'].isnull()]['location'].unique())

# population_density may be an important features for a diseas like a virus,
# but there are way too many location with NaN values, so we cannot replace them
# with 0s nor infere a mean.
# let's drop it

owid.drop('population_density', axis=1, inplace=True)

# many more columns remain with NaN values.
# these cannot be treated as the first columns, so 0s aren't logically exhaustive.

print(owid[owid['gdp_per_capita'].isnull()]['location'].unique())
print(owid[owid['location'] = 'Faeroe Islands']['gdp_per_capita'])
print(owid[owid['location'] = 'Vatican']['gdp_per_capita'])
print(owid[owid['location'] = 'Monaco']['gdp_per_capita'])

# we see that none of the rows are filled with truthy values, so methods like
# bfill and ffill are useless.
# indeed a consideration can be made.
# for this uni project, precision in data mining can be not that precious, so
# speicifically for columns like handwashing_facilities, human_development index
# and stringency_index (just to name a few), NaN values can be replaced with the
# mean of the same features, grouped by the location's continent.

print(owid('continent')['female_smokers'].mean())
continents = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America', 'World']

# defined a function that takes the column, creates a list of the means of
# that column based on the mean for the associated continent
# finally loops through the list and every row of the dataset and if the
# the continent matches AND the value is NaN, it replaces it with the mean

def filling_the_na_values(column_name):
  list_of_means = owid.groupby('continent')[column_name].mean().tolist()
  zipped_list = [(x, y) for x, y in zip(continents, list_of_means)]

  for i in zipped_list:
    for index in owid.index:
      if owid.loc[index, 'continent'] == i[0] and np.isnan(owid.loc[index, column_name]):
        owid.loc[index, column_name] = i[1]

# although the expensive loops, for some reason the code below (which takes advantage
# of pandas datframe.loc property), didn't fill the values, although it could access
# the column (series) efficiently.

# THE CODE:
# def filling_the_na_values_with_means(column):
#   list_of_means = owid.groupby('continent')[column].mean().tolist()
#   zipped_list = [(x, y) for x, y in zip(continents, list_of_means)]
#
#   for i in zipped_list:
#     # owid.iloc[owid['continent'] == i[0] & owid[column]].fillna(i[1], inplace=True)
#     owid.loc[owid['continent'] == i[0], column].fillna(i[1], inplace=True)

filling_the_na_values_with_means('stringency_index')

# some data exploration

import matplotlib.pyplot as plt
import seaborn as sb

owid.hist(column=owid.columns[3:12], bins=100, figsize=(10,10))
owid[owid['continent'] != 'World']['total_deaths'].hist(bins=100)
total_cases_max_mask = owid.groupby('location')['total_cases'].max()
total_deaths_max_mask = owid.groupby('location')['total_deaths'].max()

plt.figure(figsize=(10, 6))
plt.xlabel('Total Deaths')
plt.ylabel('Total Cases')
plt.scatter(
    total_deaths_max_mask,
    total_cases_max_mask,
    c=range(len(owid['location'].unique()))
    )

# the plots seem biased by the high value of the rows with 'continent' == 'World'
# we shall create another DF with the World continent value, which we name 'macro_owid'

macro_owid = owid[owid['continent'] == 'World']
owid.drop(owid[owid['continent'] == 'World'].index, inplace=True)
