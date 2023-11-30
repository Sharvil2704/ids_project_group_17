import pandas as pd

# Load the Breast Cancer dataset
url = 'https://archive.ics.uci.edu/static/public/15/data.csv'
df = pd.read_csv(url)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Display attribute information and statistics
print("Attribute Information and Statistics:")
print(df.describe())

# Display null values in the dataset
print("Null values in the dataset:")
print(df.isnull().sum())

# Visualize the dataset in the form of a table
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.expand_frame_repr', False)  # Allow the DataFrame to be displayed on one page
print("Visual representation of the dataset:")
print(df)
