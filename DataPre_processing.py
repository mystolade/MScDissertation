#!/usr/bin/env python
# coding: utf-8

# In[5]:


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



# In[ ]:




