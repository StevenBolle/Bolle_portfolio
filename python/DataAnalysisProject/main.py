import pandas as pd
import matplotlib.pyplot as plt
from predictor import Predictor

# Wael Mujali contribution
file_path = 'violationTimeOfYear.csv'
df = pd.read_csv(file_path)


df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Month'] = df['Date'].dt.month

print("First 10 rows of the dataset:\n")
print(df.head(10))

print("\nColumn Data Types:\n")
print(df.dtypes)

plt.figure(figsize=(12, 8))
df['Month'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Month')
plt.ylabel('Number of Violations')
plt.title('Total Violations per Month')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))
plt.scatter(df['Month'], df['Age'], alpha=0.5)
plt.xlabel('Month')
plt.ylabel('Age')
plt.title('Age of Violators by Month')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
for sex in df['Sex'].dropna().unique():
    subset = df[df['Sex'] == sex]
    plt.scatter(subset['Month'], subset.index, label=sex, alpha=0.6)

plt.xlabel('Month')
plt.ylabel('Index')
plt.title('Violations by Sex Across Months')
plt.legend(frameon=False)
plt.grid(True)
plt.tight_layout()
plt.show()

# End WM contribution

# -------------------------------------------------------------------------------------------------------
# line plot by SMB --------------------------------------------------------------------------------------
df = pd.read_csv('violationTimeOfYear.csv')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# helper method for filtering data
def valid_string(s):
    if pd.isna(s):
        return False
    stripped = s.strip()
    if stripped == '' or stripped.lower() == 'unknown':
        return False
    return True

clean_gend = df['Sex'].apply(valid_string)
clean_desc = df['Description'].apply(valid_string)

# filter data
df_clean = df[df['Date'].notna() & df['Age'].notna() & clean_gend & clean_desc].copy()
df_speed = df_clean[df_clean['Description'].str.lower().str.contains('speed', na=False)].copy()

# create month column from data in 'Date' column
df_speed['Month'] = df_speed['Date'].dt.month

# use new 'Month' column and 'Sex' column to group and total with
month_count = df_speed.groupby(['Month','Sex']).size().reset_index(name='Count')
plt.figure(figsize=(8,6))
for sex in ['Male', 'Female']:
    data = month_count[month_count['Sex'] == sex]
    plt.plot(data['Month'], data['Count'], marker='o', linewidth=2, label=sex, color='red' if sex =='Male' else 'blue')
plt.title('Speeding Violations by Month', fontsize=20, pad=15, fontweight='semibold', color='black')
plt.xlabel('Month', fontsize=16, labelpad=10, color='black')
plt.ylabel('Number of Violations', fontsize=16, labelpad=10, color='black')
plt.xticks(range(1,13), range(1,13), color='black')
plt.yticks(color='black')

# https://matplotlib.org/2.0.2/api/pyplot_api.html#matplotlib.pyplot.gca referenced for labeling using .gca()
ax = plt.gca()
ax.grid(axis='y', color='white')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible('black')
ax.spines['bottom'].set_visible('black')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# end line plot ---------------------------------------------------------------------------------------------

# second line chart without NYPD data
# to make NYPD have datetime64[ns] - had to merge month, year, arbitrary day of 01 added to all dates from NYPD
df_speed = df_speed[df_speed['Date'].dt.day != 1]

plt.figure(figsize=(10,8))

for sex in ['Male', 'Female']:
    data = month_count[month_count['Sex'] == sex]
    plt.plot(data['Month'], data['Count'], marker='o', linewidth=2, label=sex, color='red' if sex =='Male' else 'blue')
plt.title('Speeding Violations by Month: Excluding NYPD', fontsize=20, pad=15, fontweight='semibold', color='black')
plt.xlabel('Month', fontsize=16, labelpad=10, color='black')
plt.ylabel('Number of Violations', fontsize=16, labelpad=10, color='black')
plt.xticks(range(1,13), range(1,13), color='black')
plt.yticks(color='black')

ax = plt.gca()
ax.grid(axis='y', color='white')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible('black')
ax.spines['bottom'].set_visible('black')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# End SMB contribution

#Daniel Cobb Contribution
predictor = Predictor()
reader = predictor.extract_months_years(file_path)
totals = predictor.get_monthly_totals(reader)

user_month = int(input("Enter a month for violations prediction(1-12): "))
user_year = int(input("Enter a year for violations prediction: "))
result, accuracy = predictor.make_predictions(totals, user_month, user_year)
print(f"Projected number of Violations for {user_month}/{user_year}: {round(result,0)}")
print(f"Accuracy score: {round(accuracy,2)}")