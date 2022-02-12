import numpy as np
import pandas as pd

url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'
owid = pd.read_csv(url)

owid_backup = owid.copy()

print(owid.head(10))

print(owid.shape)

print(owid.info())

print(owid.describe())
