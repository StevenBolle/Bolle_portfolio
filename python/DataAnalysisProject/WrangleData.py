from operator import index
import pandas as pd
import numpy as np
import seaborn as sbn
from datetime import datetime
from pandas import read_csv

# display options for testing
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# class to clean data and make it uniform across csv files
# SMB referenced for filtering https://youtu.be/bDhvCp3_lYw?si=AI7Ryhi8o6YdxD_S
# SMB referenced for parsing dates https://youtu.be/RO6WIGX21NI?si=o_TwF6IdHKetCgs0
# SMB referenced for manipulating date/time columns https://pandas.pydata.org/docs/user_guide/timeseries.html
class DataCleaning:
    def __init__(self, df):
        self.df = df.copy()

    def clean_date_column(self, date_column, output_format='%m/%d/%y'):
        # separate date and time from date column, make date format uniform
        self.df[date_column] = self.df[date_column].astype(str).str.replace(' 0:00', '', regex=False).str.strip()
        # date time convert
        self.df[date_column] = pd.to_datetime(self.df[date_column], format='%m/%d/%Y')
        # format the output
        #self.df[date_column] = self.df[date_column].dt.strftime(output_format)
        return self.df

    def ny_clean_date_column(self,date_column, output_format='%m/%d/%y'):
        if date_column not in self.df.columns and 'Year' in self.df.columns and 'Violation Month' in self.df.columns:
            self.df[date_column] = self.df.apply(lambda row: '{:02d}/01/{}'.format(
                int(row['Violation Month']),
                int(row['Year'])
            ), axis =1)
            self.df[date_column] = pd.to_datetime(self.df[date_column], format='%m/%d/%Y')
        else:
            self.df[date_column] = pd.to_datetime(self.df[date_column], format='%m/%d/%y')
        return self.df

    def standardize_time_column(self, time_column):
        # make all time information 24 hour standard
        def to_24hr(time):
            time_string = str(time).strip().lower()

            if time_string == 'nan' or time_string == '' or time_string == 'none' or time_string == 'NaN':
                return None
            time_format_to_try = ['%I:%M%p','%H:%M:%S','%H:%M']
            for time_format in time_format_to_try:
                try:
                    dt_obj = datetime.strptime(time_string, time_format)
                    return dt_obj.strftime('%H:%M')
                except ValueError:
                    continue
            return None
        self.df[time_column] = self.df[time_column].apply(to_24hr)
        return self.df

    def split_date_time_column(self, date_time_column, date_column='Date', time_column='Time'):
        self.df[date_time_column] = pd.to_datetime(self.df[date_time_column], errors='coerce')
        # parse and extract date and time to new columns
        self.df[date_column] = self.df[date_time_column].dt.strftime('%m/%d/%Y')
        self.df[time_column] = self.df[date_time_column].dt.strftime('%H:%M')

        return self.df

    #austinTX filtering (and other applicable csv file columns)
    def filter_by_offense_type(self, offense_type_column, disallowed_types):
        self.df = self.df[~self.df[offense_type_column].isin(disallowed_types)]
        return self.df

    def filter_by_column(self, column, value, contains=True):
        if contains:
            self.df = self.df[self.df[column].str.contains(value, na=False)]
        else:
            self.df = self.df[self.df[column] == value]
        return self.df

    # replace values that will make it difficult to aggregate later
    def standardize_gender(self, sex_column):
        mapping = {
            'M':'Male',
            'Male':'Male',
            'MALE':'Male',
            'F':'Female',
            'Female':'Female',
            'FEMALE':'Female',
            'U':'Unknown',
            'Unknown':'Unknown',
            'UNKNOWN':'Unknown',
            'NaN':'Unknown',
            'Nan':'Unknown',
            'nan':'Unknown',
            None:'Unknown'
        }
        self.df[sex_column] = self.df[sex_column].fillna('Unknown')
        self.df[sex_column] = self.df[sex_column].replace(mapping)
        return self.df


    def standardize_race(self, race_column):
        mapping = {
            'W':'White',
            'White':'White',
            'WHITE':'White',
            'White non Spanish descent':'White',
            'A':'Asian',
            'Asian':'Asian',
            'ASIAN':'Asian',
            'Asian or pacific islander':'Asian',
            'H':'Hispanic',
            'Hispanic':'Hispanic',
            'HISPANIC':'Hispanic',
            'Hispanic or Latino':'Hispanic',
            'Other':'Unknown',
            'O':'Unknown',
            'U':'Unknown',
            'UNKNOWN':'Unknown',
            'B':'Black',
            'Black':'Black',
            'Black or African American':'Black',
            'BLACK':'Black',
            'Native American':'Native American',
            'Alaskan':'Native American',
            'Native American or Alaskan':'Native American',
            'Middle Eastern':'Middle Eastern',
            'unknown':'Unknown',
            'Other':'Unknown',
            'Nan':'Unknown',
            'Nan':'Unknown',
            'nan':'Unknown',
            None:'Unknown',
            'L':'Hispanic',
            'M':'Hispanic',
            'I':'Native American',
            'IB': "Unknown",
            'K': "Unknown"
        }
        self.df[race_column] = self.df[race_column].fillna('Unknown')
        self.df[race_column] = self.df[race_column].replace(mapping)
        return self.df

    def clean_string_columns(self):
        for column in self.df.select_dtypes(include=['object']).columns:
            self.df[column] = self.df[column].str.strip()
        return self.df

    def filter_violation_type(self, violation_col, violation_type):
        self.df = self.df[self.df[violation_col].str.lower() == violation_type.lower()]
        return self.df

# *******************************************************************************************************
# *******************************************************************************************************
# *******************************************************************************************************

# austin TX - cleaned! ----------------------------------------------------------
austinTX = pd.read_csv('austinTX.csv')
print("Austin Texas data head: \n")
print(austinTX.dtypes)
print(austinTX.head(10))
filter_austin = austinTX[~austinTX['Offense_Type'].isin(['PK', 'CP','OR'])]
austin_cleaner = DataCleaning(filter_austin)
cleaned_austin = austin_cleaner.df
austin_cleaner.clean_date_column('Date')
austin_cleaner.standardize_time_column('Time')
austin_cleaner.standardize_race('Race')
austin_cleaner.standardize_gender('Sex')
cleaned_austin = austin_cleaner.df
cleaned_austin.to_csv('newAustinTX.csv', index=False)
print('\nCleaned and filtered austin TX dat\n')
print(cleaned_austin.head(10))

# albanyNY cleaned!
albanyNY = pd.read_csv('albanyNY.csv')
print('Albany data before cleaning: \n')
print(albanyNY.head(10))
albany_cleaner = DataCleaning(albanyNY)
albany_clean = albany_cleaner.standardize_gender('Sex')
albany_clean = albany_cleaner.clean_date_column('Date')
albany_clean = albany_cleaner.standardize_time_column('Time')
print('\nAlbany data after cleaning: \n')
print(albany_clean.head(10))
albany_clean.to_csv('newAlbanyNY.csv', index=False)

# cedarLakeIN - cleaned and working
cedarlakeIN = pd.read_csv('cedarLakeIN.csv')
print('\nCedar Lake IN before cleaning: \n')
print(cedarlakeIN.head(10))
cedar_cleaner = DataCleaning(cedarlakeIN)
cedarlakeIN = cedar_cleaner.clean_date_column('Date')
cedarlakeIN = cedar_cleaner.standardize_time_column('Time')
cedarlakeIN = cedar_cleaner.standardize_gender('Sex')
cedarlakeIN = cedar_cleaner.standardize_race('Race')
print('\nCedar Lake after cleaning: \n')
cedarlakeIN.to_csv('newCedarLakeIN.csv', index=False)
print(cedarlakeIN.head(10))

# coloradoSpringsCO - cleaned and working
coloradoSpringsCO = pd.read_csv('coloradoSpringsCO.csv')
print('\nColorado Springs CO before cleaning: \n')
print(coloradoSpringsCO.dtypes)
print(coloradoSpringsCO.head(10))
colorado_cleaner = DataCleaning(coloradoSpringsCO)
colorado_cleaner.split_date_time_column('Date_Time')
colorado_cleaner.df.drop(columns=['Date_Time'], inplace=True)
colorado_cleaner.filter_by_column('Offense_Type', 'Traffic')
colorado_cleaner.clean_date_column('Date')
colorado_cleaner.standardize_time_column('Time')
cleanedColorado = colorado_cleaner.df
cleanedColorado.to_csv('newColoradoSpringsCO.csv', index=False)
print('\nColorado springs data cleaned and save to newColoradoSpringsCO\n')
new_colorado_springs = pd.read_csv('newColoradoSpringsCO.csv')
print(new_colorado_springs.head(10))

# griffithIN - cleaned, working
griffith = pd.read_csv('griffithIN.csv')
print('\nGriffith IN before cleaning: \n')
print(griffith.head(10))
grifclean = DataCleaning(griffith)
griffith = grifclean.standardize_race('Race')
griffith = grifclean.clean_date_column('Date')
griffith = grifclean.standardize_time_column('Time')
griffith = grifclean.standardize_gender('Sex')
print('\nGriffith IN after cleaning: \n')
print(griffith.head(10))
griffith.to_csv('newGriffithIN.csv', index=False)
print('newGriffithIN.csv created')

# louisvilleKY - cleaned
louisville = pd.read_csv('LouisvilleKY.csv')
print('\nlouisvilleKY before cleaning: \n')
print(louisville.head(10))
louclean = DataCleaning(louisville)
louisville = louclean.clean_string_columns()
louisville = louclean.split_date_time_column('Date_Time')
louisville = louclean.clean_date_column('Date')
louisville = louclean.standardize_time_column('Time')
louisville = louclean.standardize_gender('Sex')
louisville = louclean.standardize_race('Race')
louisville = louisville[['Date', 'Time', 'Age', 'Sex', 'Race', 'Ethnicity', 'Description']]
print('\nLouisville KY data after cleaning: \n')
print(louisville.head(10))
louisville.to_csv('newLouisvilleKY.csv', index=False)
print('Cleaned data saved to newLouisvilleKY.csv')

# montgomeryMD - cleaned
print('MontgomeryMD data: \n')
monty = pd.read_csv('montgomeryMD.csv')
moclean = DataCleaning(monty)
monty = moclean.filter_violation_type('Violation Type', 'citation')
monty = moclean.standardize_race('Race')
monty = moclean.standardize_gender('Sex')
monty = moclean.clean_date_column('Date')
monty = moclean.standardize_time_column('Time')
print('\nData cleaned \n')
print(monty.head(10))
monty.to_csv('newMontgomeryMD.csv')
print('\nnewMontgomeryMD.csv created')

# NYPD - cleaned
newyork = pd.read_csv('NYPD.csv')
newyork['Date'] = newyork.apply(lambda row: '{:02d}/01/{:02d}'.format(int(row['Violation Month']), int(str(row['Year'])[-2:])), axis=1)
print(newyork.head(3))
newclean = DataCleaning(newyork)
newyork = newclean.ny_clean_date_column('Date')
newyork = newclean.standardize_gender('Sex')
newyork = newyork.drop(columns=['Violation Month', 'Violation Day of Week', 'Year'])
print('\nNYPD data cleaned!\n')
print(newyork.head(3))
newyork.to_csv('newNYPD.csv', index=False)

# somervilleMA - cleaned
df = pd.read_csv('somervilleMA.csv')
print('Somerville MA loaded\n')
print(df.head(3))
df['Date and Time Issued '] = df['Date and Time Issued '].str.strip()
df['Time'] = df['Date and Time Issued '].str.split(' ').str[1]
df['Date'] = df['Date and Time Issued '].str.split(' ').str[0]
cleanit = DataCleaning(df)
df = cleanit.clean_date_column('Date')
df = cleanit.standardize_time_column('Time')
df = df.drop(columns=['Date and Time Issued '])
df = df[['Date', 'Time', 'Violation Description']]
print('somervilleMA cleaned\n')
print(df.head(3))
df.to_csv('newSomervilleMA.csv', index=False)
print('\nnewSomervilleMA.csv created')

# cincinattiOH - cleaned!
df = read_csv('cincinattiOH.csv')
print('cincinattiOH loaded')
print(df.head(20))
# filter out unnecessary rows
df = df[df['FIELD_SUBJECT'] == 'DRIVER']
df['Time'] = df['DATE'].str.split(' ').str[1]
df['Date'] = df['DATE'].str.split(' ').str[0]
cleancin = DataCleaning(df)
df = cleancin.clean_date_column('Date')
df = cleancin.standardize_time_column('Time')
df = cleancin.standardize_gender('SEX')
df = cleancin.standardize_race('RACE')
df = df.drop(columns=['DATE', 'FIELD_SUBJECT'])
df = df[['Date', 'Time', 'SEX', 'RACE', 'AGE_RANGE']]
df.columns = df.columns.str.title()
df = df.drop_duplicates()
print('\nCincinatti OH cleaned\n')
print(df.head(20))
df.to_csv('newCincinattiOH.csv', index=False)

# phoenixAZ
df = pd.read_csv('phoenixAZ.csv')
print('phoenixAZ loaded! \n')
df.columns = df.columns.str.title()
bird = DataCleaning(df)
# date column is already in datetime64[ns] , no need to clean
df = bird.standardize_gender('Sex')
df = bird.standardize_race('Race')
df = bird.standardize_gender('Issuing_Officer_Sex')
df = bird.standardize_race('Issuing_Officer_Race')
df['Day_Of_Week'] = df['Day_Of_Week'].str.split('-').str[1]
print('\nPhoenix cleaned!\n')
print(df.head(10))
df.to_csv('newPhoenixAZ.csv', index=False)