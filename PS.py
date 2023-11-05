

import pandas as pd

# Read the .txt file into a Pandas DataFrame
df = pd.read_csv('/Users/ybharadwaj2/Downloads/property_locations.txt', delimiter='\t',encoding='cp1252')

# train = pd.read_csv('/Users/ybharadwaj2/Downloads/property_locations.txt')
# Print the DataFrame
df.head()

df.columns

df.info()

(df.isna().sum())*100/df.shape[0]
# 5.6% of dataset have nulls

df.shape

# unique count
df.nunique()

df.PropertyStatus.value_counts()


