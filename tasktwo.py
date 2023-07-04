from IPython.display import display
import pandas as pd

path = "/Users/ogvie/Desktop/Cognizant/sales.csv"
sales_df = pd.read_csv(path)
sales_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
sales_df.head()
display(sales_df)

path = "/Users/ogvie/Desktop/Cognizant/sensor_stock_levels.csv"
stock_df = pd.read_csv(path)
stock_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
stock_df.head()
display(stock_df)

path = "/Users/ogvie/Desktop/Cognizant/sensor_storage_temperature.csv"
temp_df = pd.read_csv(path)
temp_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
temp_df.head()
display(temp_df)

def convert_to_datetime(data: pd.DataFrame = None, column: str = None):

  dummy = data.copy()
  dummy[column] = pd.to_datetime(dummy[column], format='%Y-%m-%d %H:%M:%S')
  return dummy

sales_df = convert_to_datetime(sales_df, 'timestamp')
stock_df = convert_to_datetime(stock_df, 'timestamp')
temp_df = convert_to_datetime(temp_df, 'timestamp')

sales_df.info()
stock_df.info()
temp_df.info()

from datetime import datetime

def convert_timestamp_to_hourly(data: pd.DataFrame = None, column: str = None):
  dummy = data.copy()
  new_ts = dummy[column].tolist()
  new_ts = [i.strftime('%Y-%m-%d %H:00:00') for i in new_ts]
  new_ts = [datetime.strptime(i, '%Y-%m-%d %H:00:00') for i in new_ts]
  dummy[column] = new_ts
  return dummy

sales_df = convert_timestamp_to_hourly(sales_df, 'timestamp')
sales_df.head()

stock_df = convert_timestamp_to_hourly(stock_df, 'timestamp')
stock_df.head()

temp_df = convert_timestamp_to_hourly(temp_df, 'timestamp')
temp_df.head()

sales_agg = sales_df.groupby(['timestamp', 'product_id']).agg({'quantity': 'sum'}).reset_index()
sales_agg.head()

stock_agg = stock_df.groupby(['timestamp', 'product_id']).agg({'estimated_stock_pct': 'mean'}).reset_index()
stock_agg.head()

temp_agg = temp_df.groupby(['timestamp']).agg({'temperature': 'mean'}).reset_index()
temp_agg.head()

merged_df = stock_agg.merge(sales_agg, on=['timestamp', 'product_id'], how='left')
merged_df.head()

merged_df = merged_df.merge(temp_agg, on='timestamp', how='left')
merged_df.head()

merged_df.info()

merged_df['quantity'] = merged_df['quantity'].fillna(0)
merged_df.info()

product_categories = sales_df[['product_id', 'category']]
product_categories = product_categories.drop_duplicates()

product_price = sales_df[['product_id', 'unit_price']]
product_price = product_price.drop_duplicates()

merged_df = merged_df.merge(product_categories, on="product_id", how="left")
merged_df.head()

merged_df = merged_df.merge(product_price, on="product_id", how="left")
merged_df.head()

merged_df.info()

display(merged_df)

merged_df['timestamp_day_of_month'] = merged_df['timestamp'].dt.day
merged_df['timestamp_day_of_week'] = merged_df['timestamp'].dt.dayofweek
merged_df['timestamp_hour'] = merged_df['timestamp'].dt.hour
merged_df.drop(columns=['timestamp'], inplace=True)
merged_df.head()
display(merged_df)

merged_df = pd.get_dummies(merged_df, columns=['category'])
merged_df.head()

merged_df.info()

merged_df.drop(columns=['product_id'], inplace=True)
merged_df.head()

X = merged_df.drop(columns=['estimated_stock_pct'])
y = merged_df['estimated_stock_pct']
print(X.shape)
print(y.shape)

K = 10
split = 0.75

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

accuracy = []

for fold in range(0, K):

  
  model = RandomForestRegressor()
  scaler = StandardScaler()

  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=42)


  scaler.fit(X_train)
  X_train = scaler.transform(X_train)
  X_test = scaler.transform(X_test)

  
  trained_model = model.fit(X_train, y_train)

 
  y_pred = trained_model.predict(X_test)

  
  mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
  accuracy.append(mae)
  print(f"Fold {fold + 1}: MAE = {mae:.3f}")

print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")

import matplotlib.pyplot as plt
import numpy as np

features = [i.split("__")[0] for i in X.columns]
importances = model.feature_importances_
indices = np.argsort(importances)

fig, ax = plt.subplots(figsize=(10, 20))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()