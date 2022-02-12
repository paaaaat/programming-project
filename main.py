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
