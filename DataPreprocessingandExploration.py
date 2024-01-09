#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# Load the Incident data
SHUdata = pd.read_csv('incident.csv')

print("Incident Overview:")
SHUdata.head(10)

# Filter out pending incidents
SHUdata = SHUdata[SHUdata['status_enum'] != 3]

# Filter out events type that are not Incidents
SHUdata = SHUdata[SHUdata['type_enum'] == 1]

# Display the modified dataset
SHUdata.head

SHUdata.shape

SHUdata.describe() 

#List of column names to drop
columns_to_drop =['csg_sc', 'inc_close_date', 'generic_cls_n', 'serv_dept_n', 'internal_id', 'inc_serious_n', 'status_enum', 'major_inc', 'type_enum']

#Drop the specified columns
SHUdata = SHUdata.drop(columns=columns_to_drop)
SHUdata.head()

# New column names
new_column_names = {
    'date_logged': 'date_logged',
    'inc_cat_n': 'incident_category',
    'inc_cause_n': 'incident_cause_category',
    'inc_major_n': 'logging_major_category',
    'inc_prior_n': 'priority_level',
    'inc_resolve_act': 'actual_resolution_date',
    'inc_resolve_due': 'proposed_resolution_date',
    'item_n': 'item_name',
    'prod_cls_n': 'product_category',
    'product_n': 'product_name',
}

# Rename the columns
SHUdata.rename(columns=new_column_names, inplace=True)

# Check data types and missing values
print("\nData Types and Missing Values:")
SHUdata.info()

#Check and display count of missing values in the dataset
print ("Missing Values")
print(SHUdata.isnull().sum())
print()

# Convert date columns to datetime

SHUdata['date_logged'] = pd.to_datetime(SHUdata['date_logged'], format="%d/%m/%Y %H:%M")
SHUdata['actual_resolution_date'] = pd.to_datetime(SHUdata['actual_resolution_date'], format="%d/%m/%Y %H:%M")
SHUdata['proposed_resolution_date'] = pd.to_datetime(SHUdata['proposed_resolution_date'], format="%d/%m/%Y %H:%M")

# Create two new columns 'SLA (hrs)' and 'SLA (days)' based on priority levels using numpy
import numpy as np

SHUdata['SLA (hrs)'] = np.select(
    [
        SHUdata['priority_level'] == 'Critical/Business Risk',
        SHUdata['priority_level'] == 'High',
        SHUdata['priority_level'] == 'Medium',
        SHUdata['priority_level'] == 'Low',
    ],
    [
        4.0,
        8.5,
        (23.5 + 42.5) / 2,
        127.3,
    ],
    default=None
)

SHUdata['SLA (days)'] = np.select(
    [
        SHUdata['priority_level'] == 'Critical/Business Risk',
        SHUdata['priority_level'] == 'High',
        SHUdata['priority_level'] == 'Medium',
        SHUdata['priority_level'] == 'Low',
    ],
    [
        0.5,
        1.0,
        (2.8 + 5.0) / 2,
        15.0,
    ],
    default=None
)

# Verify the changes
print(SHUdata[['priority_level', 'SLA (hrs)', 'SLA (days)']])


# Impute missing values in proposed_resolution_date based on 'date_logged' and 'SLA (hrs)'
# Impute missing values in 'proposed_resolution_date' based on 'date_logged' and 'SLA (hrs)'
SHUdata['proposed_resolution_date'] = SHUdata.apply(
    lambda row: row['date_logged'] + pd.Timedelta(hours=row['SLA (hrs)']) if pd.isnull(row['proposed_resolution_date']) else row['proposed_resolution_date'],
    axis=1
)

# Verify the changes
print(SHUdata[['date_logged', 'SLA (hrs)', 'proposed_resolution_date']])


#Check and display count of missing values in the dataset
print ("Missing Values")
print(SHUdata.isnull().sum())
print()


SHUdata.describe().transpose()

# Convert 'date_logged' to datetime if not already in datetime format
SHUdata['date_logged'] = pd.to_datetime(SHUdata['date_logged'])

# Set the end date for the time series analysis (August 31, 2023)
end_date = pd.to_datetime('2023-08-31')

# Filter data to include only entries until the end date
SHUdata_filtered = SHUdata[SHUdata['date_logged'] <= end_date]

    
# Calculate the difference in days and create a new column 'days_to_resolve'
SHUdata['days_to_resolve'] = (SHUdata['actual_resolution_date'] - SHUdata['date_logged']).dt.days

# Display the updated DataFrame with the new column
SHUdata.head()
    

# Group by priority level and calculate descriptive statistics
grouped_stats = SHUdata.groupby('priority_level')['days_to_resolve'].describe()

# Display the statistics table
print("\nDescriptive Statistics for Resolution Time Across Priority Levels:")
print(grouped_stats)

# Select rows where resolution time is negative

# Filter rows with negative 'days_to_resolve' and display the top 10
top10_negative_days_rows = SHUdata[SHUdata['days_to_resolve'] < 0].head(10)

# Print the filtered DataFrame
print(top10_negative_days_rows[['IncidentID','date_logged', 'priority_level','actual_resolution_date', 'days_to_resolve', 'proposed_resolution_date',  'SLA (hrs)', 'SLA (days)']])

# Filter rows with negative 'days_to_resolve'
negative_days_rows = SHUdata[SHUdata['days_to_resolve'] < 0]

# write to csv
csv_file_path = 'negative_days_data.csv'

# Write the DataFrame to a CSV file
negative_days_rows.to_csv(csv_file_path, index=False)

# Print a message indicating the file has been saved
print(f"The data with negative 'days_to_resolve' values has been saved to {csv_file_path}.")

#rewrite the dataframe to show only filtered rows tht do not have negative days_to_resolve field i.e where resolution date is less than the date logged
SHUdata = SHUdata[SHUdata['days_to_resolve'] >= 0]

plt.figure(figsize=(10, 6))
sns.boxplot(x='priority_level', y='days_to_resolve', data=SHUdata)
plt.title('Resolution Time Distribution Across Priority Levels')
plt.xlabel('Priority Level')
plt.ylabel('Days to Resolve')
plt.show()

# Group by priority level and calculate descriptive statistics
grouped_stats = SHUdata.groupby('priority_level')['days_to_resolve'].describe()

# Display the statistics table
print("\nDescriptive Statistics for Resolution Time Across Priority Levels:")
print(grouped_stats)

#display scatterplot
plt.figure(figsize=(12, 6))
sns.scatterplot(x='IncidentID', y='days_to_resolve', data=SHUdata, alpha=0.5)
plt.title('Incident Volume vs. Resolution Time')
plt.xlabel('Number of Incidents')
plt.ylabel('Days to Resolve')
plt.show()


# Group by days_to_resolve and count incidents
incident_counts = SHUdata.groupby('days_to_resolve')['IncidentID'].count().reset_index()

# Display the table
print("\nCount of Incidents Against Resolution Time in Days:")
print(incident_counts)

# Main Data Preprocessing
# Set 'date_logged' as the index for time series analysis
#this is to enable time series analysis by setting the datelogged column as the index.
SHUdata.set_index('date_logged', inplace=True)

# Handle missing values (if any)
# Fill missing values in the target variable (days_to_resolve) with the mean of the column
SHUdata['days_to_resolve'].fillna(SHUdata['days_to_resolve'].mean(), inplace=True)

SHUdata_resampled = SHUdata

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply Label Encoding to each column
for column in SHUdata_resampled.columns:
    SHUdata_resampled[column] = label_encoder.fit_transform(SHUdata_resampled[column])

print(SHUdata_resampled)

#Creating a heatmap of the correlations between the numeric columns of the 'SHUdata' DataFrame
plt.figure(figsize=(18,8))

training_SHUdata = SHUdata.corr()

mask = np.triu(np.ones_like(training_SHUdata))
plt.title('Correlation of numerical features', y=1.05, size=15)
sns.heatmap(training_SHUdata, annot=True, cbar=False, mask=mask)
plt.show()
print(SHUdata.shape)

print(len(SHUdata_resampled))
print(len(SHUdata))

from scipy.stats import chi2_contingency

# Assuming 'data' is your DataFrame containing categorical variables
# Replace 'variable1' and 'variable2' with the actual column names
contingency_table = pd.crosstab(SHUdata['incident_category'], SHUdata['priority_level'])

# Display the contingency table
print("\nContingency Table:")
print(contingency_table)

# Perform the chi-squared test
chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)

# Display the chi-squared statistic and p-value
print(f"\nChi-squared Statistic: {chi2_stat}")
print(f"P-value: {p_value}")

SHUdata_resampled['day_of_week'] = SHUdata_resampled.index.dayofweek
SHUdata_resampled['month'] = SHUdata_resampled.index.month
SHUdata_resampled['quarter'] = SHUdata_resampled.index.quarter

# Use Min-Max scaling to scale the target variable ('days_to_resolve') to a range between 0 and 1
scaler = MinMaxScaler()
SHUdata_resampled['days_to_resolve_scaled'] = scaler.fit_transform(SHUdata_resampled[['days_to_resolve']])

# Print the preprocessed DataFrame
SHUdata_resampled.head()

# Function to split data into train and test sets
def train_test_split_data(data, test_size=0.2):
    split_index = int(len(data) * (1 - test_size))
    train_data, test_data = data[:split_index], data[split_index:]
    return train_data, test_data


# In[ ]:




