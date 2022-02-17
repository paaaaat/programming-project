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

# much better
# the iso_code it's not that important

owid.drop('iso_code', axis=1, inplace=True)

# strange to read that continent misses some values
# we can see that 'location' for NaN 'continents' are either continents
# themselves, income classification and World
# we can say that their 'continent' can be 'World'

owid['continent'].fillna(value='World', inplace=True)

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

# now it's time to replace the NaN values
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

# let's fix the population

owid.loc[owid['location'] == 'International', 'population'] = 7900000000
owid.loc[owid['location'] == 'Northern Cyprus', 'population'] = 326000

# population_density may be an important features for a diseas like a virus,
# but there are way too many location with NaN values, so we cannot replace them
# with 0s nor infere a mean.
# let's drop it

owid.drop('population_density', axis=1, inplace=True)

# many more columns remain with NaN values.
# these cannot be treated as the first columns, so 0s aren't logically exhaustive.
# we see that none of the rows are filled with truthy values, so methods like
# bfill and ffill are useless.
# indeed a consideration can be made.
# for this project, precision in data mining can be not that precious, so
# specifically for columns like handwashing_facilities, human_development index
# and stringency_index (just to name a few), NaN values can be replaced with the
# mean of the same features, grouped by the location's continent.

continents = owid['continent'].unique()

# defined a function that takes the column, creates a list of the means of
# that column based on the mean for the associated continent
# finally loops through the list and every row of the dataset and if the
# the continent matches AND the value is NaN, it replaces it with the mean

def filling_the_na_values_with_means(column_name):
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
#     owid.loc[owid['continent'] == i[0], column].fillna(i[1], inplace=True)

for column in owid.columns[index_stringency_index:]:
    filling_the_na_values_with_means(column)

# some data exploration

import matplotlib.pyplot as plt
import seaborn as sb

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
plt.show()

# the plots seem biased by the high value of the rows with 'continent' == 'World'
# we shall create another DF with the World continent value, which we name 'macro_owid'

macro_owid = owid[owid['continent'] == 'World']
owid.drop(owid[owid['continent'] == 'World'].index, inplace=True)

# as our intention is not to plot time series, we can drop all columns starting
# with 'new', as they are grouped in the 'total' columns

owid.drop(
    labels=[x for x in owid.columns if 'new' in x],
    axis=1,
    inplace=True
    )

# so our new dataframe will consist of the total and fixed values: we can
# take the rows that have 01/01/2022

owid_backup = owid.copy()
owid = owid[owid['date'] == '2022-01-01']
owid.reset_index(drop=True, inplace=True)
owid.drop(labels='date', axis=1, inplace=True)

total_cases_mask = owid.groupby('continent')['total_cases'].sum()

# some modifications to Seaborn and matplotlib

sns.set_style('darkgrid')
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=13)
plt.rc('legend', fontsize=13)
plt.rc('font', size=13)

# let's see the correlations

plt.figure(figsize=(18, 13), tight_layout=True)
sns.heatmap(owid.corr(), annot=True, cmap='viridis')
plt.show()

# cases per covid disease by continent

plt.figure(figsize=(8,6), tight_layout=True)
colors = sns.color_palette('pastel')
plt.bar(owid['continent'].unique(), total_cases_mask, color=colors[:5])
plt.xlabel('Continent')
plt.ylabel('Total Cases')
plt.title('Cases per Covid disease by continent')
plt.show()

# The continent with the most deaths per Covid seems to be Europe.
# The stringency level is correlated with the cases per million inhabitants?

plt.figure(figsize=(13, 10), tight_layout=True)
ax = sns.scatterplot(data=owid, x='total_cases_per_million', y='stringency_index', hue='continent', palette='pastel', s=60)
ax.set(xlabel='Total Cases per Million', ylabel='Stringency Index')
ax.legend(title='Continent', title_fontsize = 12)
plt.show()

# can't tell
# what about the total cases per million inhabitans with the median age?

plt.subplots(figsize=(10, 10))
sns.scatterplot(data=owid, x='total_cases_per_million', y='median_age', s=5, color=".15")
sns.histplot(data=owid, x='total_cases_per_million', y='median_age', bins=50, pthresh=.1, cmap="mako")
sns.kdeplot(data=owid, x='total_cases_per_million', y='median_age', levels=5, color="w", linewidths=1)

# seems that very young countries have suffered less cases
# we use cases and deaths per million beacuse cut the outliers
# e.g. USA have by far the most deaths in total, but not the most deaths per million

plt.figure(figsize=(20, 8))
plt.xlabel('Total Deaths')
plt.ylabel('Total Cases')

plt.subplot(1, 2, 1)
plt.scatter(
    owid[owid['continent'] == 'North America']['total_deaths'],
    owid[owid['continent'] == 'North America']['total_cases'],
    c=range(len(owid[owid['continent'] == 'North America']['location'].unique())))

plt.subplot(1, 2, 2)
plt.scatter(
    owid[owid['continent'] == 'North America']['total_deaths_per_million'],
    owid[owid['continent'] == 'North America']['total_cases_per_million'],
    c=range(len(owid[owid['continent'] == 'North America']['location'].unique())))

# so it seems reasonable to work with per_million
# from the graph below, we see that continents with the highest median age
# have suffered the most Deaths
# i.e. Europe, which has the highest median age versus Africa, a very young continent

plt.figure(figsize=(13, 10), tight_layout=True)

ax = sns.scatterplot(
    data=owid,
    x='total_deaths_per_million',
    y='median_age',
    hue='continent',
    palette='pastel',
    s=60
    )

ax.set(xlabel='Total Deaths per Million', ylabel='Median Age')
ax.legend(title='Continent', title_fontsize = 12)
