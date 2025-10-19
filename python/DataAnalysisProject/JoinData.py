from operator import index
import pandas as pd
import numpy as np
import seaborn as sbn
from datetime import datetime
from functools import reduce

# display options for testing
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# concatenate files and create 'master csv' files to use for data viz / machine learning

# first file == 'violationTimeOfYear.csv' - created
# SMB - referenced for concat() https://youtu.be/HRzthi_FqAA?si=5oTSqfVM0QigUYkJ
files = ['newAlbanyNY.csv', 'newSomervilleMA.csv', 'newMontgomeryMD.csv', 'newNYPD.csv', 'newLouisvilleKY.csv',
         'newColoradoSpringsCO.csv', 'newCedarLakeIN.csv', 'newGriffithIN.csv', 'newAustinTX.csv']
frames = []
for file in files:
    df = pd.read_csv(file)

    # drop unwanted columns
    col_to_drop = ['Offense_Type', 'Ethnicity', 'Accident', 'Violation Type', 'Unnamed: 0']
    df = df.drop(columns=[column for column in col_to_drop if column in df.columns])
    df['Date'] = pd.to_datetime(df['Date'])

    if 'Violation Description' in df.columns:
        df = df.rename(columns={'Violation Description': 'Description'})

    frames.append(df)

all_violations = pd.concat(frames, ignore_index=True)
all_violations.to_csv('violationTimeOfYear.csv', index=False)

print('Column heads: ', all_violations.columns.tolist())
print('\nDataset shape: ', all_violations.shape)
print('\nUnique values in race column: \n')
racounts = df['Race'].value_counts(dropna=False)
print(racounts)
print('total of unique race counts: ', len(racounts))
print('\nviolationTimeOfYear head case: \n')
print(all_violations.head(10))
print(all_violations.dtypes)



#*********************************************************************************************************

# second file == 'profileOfRaceCitations.csv' - Created!
# racial profile of traffic citation
files = ['newCincinattiOH.csv', 'newCedarLakeIN.csv', 'newAustinTX.csv', 'newMontgomeryMD.csv',
        'newPhoenixAZ.csv', 'newNYPD.csv', 'newLouisvilleKY.csv', 'newGriffithIN.csv',
         'newColoradoSpringsCO.csv']

frames = []
for file in files:
    df = pd.read_csv(file)

    # drop unwanted columns
    col_to_drop = ['Offense_Type', 'Accident', 'Violation Type', 'Unnamed: 0', 'Age_Range', 'Day_Of_Week', 'Civil_Traffic_Violations', 'Criminal_Traffic_Violations']
    df = df.drop(columns=[column for column in col_to_drop if column in df.columns])
    df['Date'] = pd.to_datetime(df['Date'])
    if 'Violation Description' in df.columns:
        df = df.rename(columns={'Violation Description': 'Description'})

    frames.append(df)

race_data = pd.concat(frames, ignore_index=True)
race_data.to_csv('profileOfRaceCitations.csv', index=False)

print('Column heads: ', race_data.columns.tolist())
print('\nDataset shape: ', race_data.shape)
print('\nUnique values in race column: \n')
racounts = df['Race'].value_counts(dropna=False)
print(racounts)
print('total of unique race counts: ', len(racounts))
print('\nprofileOFRaceCitations head case: \n')
print(race_data.head(10))
print(race_data.dtypes)



#**************************************************************************************************************

# third file == 'citationAgeDemographics.csv' - created!
# age profile of traffic citation

files = ['newCedarLakeIN.csv', 'newAlbanyNY.csv', 'newPhoenixAZ.csv',
         'newNYPD.csv', 'newLouisvilleKY.csv', 'newGriffithIN.csv',
         'newColoradoSpringsCO.csv']

frames = []
for file in files:
    df = pd.read_csv(file)

    # drop unwanted columns
    col_to_drop = ['Offense_Type', 'Accident', 'Violation Type', 'Unnamed: 0', 'Age_Range',
                   'Day_Of_Week', 'Civil_Traffic_Violations', 'Criminal_Traffic_Violations']
    df = df.drop(columns=[column for column in col_to_drop if column in df.columns])
    df['Date'] = pd.to_datetime(df['Date'])

    if 'Violation Description' in df.columns:
        df = df.rename(columns={'Violation Description': 'Description'})

    frames.append(df)

age_data = pd.concat(frames, ignore_index=True)
age_data.to_csv('citationAgeDemos.csv', index=False)

print('Column heads: ', age_data.columns.tolist())
print('\nDataset shape: ', age_data.shape)
print('\ncitationAgeDemos head case: \n')
print(age_data.head(10))
print(age_data.dtypes)

