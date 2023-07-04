import pandas as pd

path = "/Users/ogvie/Desktop/Cognizant/sample_sales_data.csv"
df = pd.read_csv(path)
df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
df.head(11)

totalMean = df["total"].mean()
paymentTypeMode = df["payment_type"].mode()[0]
quantityMean = df["quantity"].mean()
customerTypeMode = df["customer_type"].mode()[0]
categoryMode = df["category"].mode()[0]
print(f"The average amount spent is £{totalMean}\n")
print(f"The most popular payment type is {paymentTypeMode}\n")
print(f"The average quantity bought is {quantityMean}\n")
print(f"The most popular customer type is {customerTypeMode}\n")
print(f"The most popular item category is {categoryMode}\n")
print(f"There are currently {len(df)} transactions on file\n")
print(f"{df.info()}\n")
print(f"{df.describe()}")

import seaborn as sns

def plot_continuous_distribution(data: pd.DataFrame = None, column: str = None, height: int = 8):
  _ = sns.displot(data, x=column, kde=True, height=height, aspect=height/5).set(title=f'Distribution of {column}');

def get_unique_values(data, column):
  num_unique_values = len(data[column].unique())
  value_counts = data[column].value_counts()
  print(f"Column: {column} has {num_unique_values} unique values\n")
  print(value_counts)

def plot_categorical_distribution(data: pd.DataFrame = None, column: str = None, height: int = 8, aspect: int = 2):
  _ = sns.catplot(data=data, x=column, kind='count', height=height, aspect=aspect).set(title=f'Distribution of {column}');

def correlation_plot(data: pd.DataFrame = None):
  corr = df.corr()
  corr.style.background_gradient(cmap='coolwarm')

plot_continuous_distribution(df, 'unit_price')
plot_continuous_distribution(df, 'quantity')
plot_continuous_distribution(df, 'total')
get_unique_values(df, 'transaction_id')
get_unique_values(df, 'product_id')
get_unique_values(df, 'category')
plot_categorical_distribution(df, 'category', height=10, aspect=3.5)
get_unique_values(df, 'customer_type')
plot_categorical_distribution(df, 'customer_type', height=5, aspect=1.5)
get_unique_values(df, 'payment_type')
plot_categorical_distribution(df, 'payment_type', height=5, aspect=1.5)
get_unique_values(df, 'timestamp')
def convert_to_datetime(data: pd.DataFrame = None, column: str = None):

  dummy = data.copy()
  dummy[column] = pd.to_datetime(dummy[column], format='%Y-%m-%d %H:%M:%S')
  return dummy
df = convert_to_datetime(df, 'timestamp')
df.info()
df['hour'] = df['timestamp'].dt.hour
df.head()
get_unique_values(df, 'hour')
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')
