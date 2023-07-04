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