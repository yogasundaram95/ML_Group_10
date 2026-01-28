#!/usr/bin/env python
# coding: utf-8

# In[43]:


import os
import pandas as pd

# Print the current working directory to verify the path
print("Current working directory:", os.getcwd())

# Load the datasets with their correct relative paths (for Excel files)
stock_signal_df = pd.read_excel('./stockSignal.xlsx')  # Correct path for stockSignal.xlsx
bdi_df = pd.read_excel('./Baltic Dry Index Historical Data.xlsx')  # Correct path for BDI data
gdp_df = pd.read_excel('./data_gpr_export.xlsx')  # Correct path for GDP data
treasury_df = pd.read_excel('./DGS10.xlsx')  # Correct path for Treasury data
news_sentiment_df = pd.read_excel('./news_sentiment_data.xlsx')  # Correct path for news sentiment data

# Display the first few rows of each dataframe to inspect
print("Stock Signal Data Preview:")
print(stock_signal_df.head())

print("BDI Data Preview:")
print(bdi_df.head())

print("GDP Data Preview:")
print(gdp_df.head())

print("Treasury Data Preview:")
print(treasury_df.head())

print("News Sentiment Data Preview:")
print(news_sentiment_df.head())


# In[44]:


# Convert 'date' columns to datetime for consistency across all datasets
stock_signal_df['date'] = pd.to_datetime(stock_signal_df['date'])
bdi_df['date'] = pd.to_datetime(bdi_df['date'], errors='coerce')  # Handle mixed formats
gdp_df['date'] = pd.to_datetime(gdp_df['date'])
treasury_df['date'] = pd.to_datetime(treasury_df['date'])
news_sentiment_df['date'] = pd.to_datetime(news_sentiment_df['date'])

# Merge the datasets one by one based on the 'date' column
merged_df = stock_signal_df
merged_df = pd.merge(merged_df, bdi_df, on='date', how='left')  # Merge with BDI data
merged_df = pd.merge(merged_df, gdp_df, on='date', how='left')  # Merge with GDP data
merged_df = pd.merge(merged_df, treasury_df, on='date', how='left')  # Merge with Treasury data
merged_df = pd.merge(merged_df, news_sentiment_df, on='date', how='left')  # Merge with News sentiment data

# Check for missing values after the merge
print(merged_df.isnull().sum())

# Handle missing values (example: forward fill)
merged_df.ffill(inplace=True)

# Display the final merged dataframe
print(merged_df.head())

# Optionally, save the merged data to a new CSV file
# merged_df.to_csv('merged_stock_data.csv', index=False)


# In[45]:


# Handle missing values (forward fill, drop, or specific methods based on your preference)
merged_df.ffill(inplace=True)  # Forward fill missing data

# Optionally drop columns with excessive missing values if not important
merged_df.drop(columns=['var_name', 'var_label'], inplace=True)  # Drop columns like var_name, var_label if they are not needed

# Optionally, you can drop rows with excessive missing values
# merged_df.dropna(thresh=100, axis=0, inplace=True)  # Keep rows with at least 100 non-NA values

# Display cleaned data
print(merged_df.head())

# Optionally, save the cleaned data to a new CSV file
# merged_df.to_csv('cleaned_merged_stock_data.csv', index=False)


# In[46]:


from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Select columns that need to be normalized (e.g., stock prices, GDP-related columns)
columns_to_normalize = ['INTC', 'ASML', 'AMAT', 'AMD', 'QCOM', 'TSM', 'TXN', 'AVGO', 'NVDA', 'GPR', 'GPRT', 'GPRH', 'DGS10']

# Apply normalization
merged_df[columns_to_normalize] = scaler.fit_transform(merged_df[columns_to_normalize])

# Display the normalized data
print(merged_df.head())


# In[47]:


# Forward-fill the missing values
merged_df.ffill(inplace=True)


# In[48]:


# Drop columns with too many missing values
merged_df.dropna(axis=1, thresh=merged_df.shape[0]*0.8, inplace=True)  # Keep columns with at least 80% non-null values


# In[49]:


# Save the final cleaned and normalized dataset
merged_df.to_csv('final_cleaned_merged_stock_data.csv', index=False)


# In[50]:


# Display the first 20 rows
display(merged_df.head(20))


# In[124]:


# Add Baltic Dry Index to main dataset
baltic_map = {date: price for row in baltic_data}
main_df['Baltic_Dry_Index'] = main_df['date'].map(baltic_map)


# In[123]:


# Feature Engineering-yosundar

#Moving Averages (7 and 30 days) for each stock
stocks = ['INTC', 'ASML', 'AMAT', 'AMD', 'QCOM', 'TSM', 'TXN', 'AVGO', 'NVDA']

for stock in stocks:
    merged_df[f'{stock}_MA7'] = merged_df[stock].rolling(window=7).mean()
    merged_df[f'{stock}_MA30'] = merged_df[stock].rolling(window=30).mean()

# Display the moving averages
print(merged_df[['date'] + [f'{stock}_MA7' for stock in stocks] + [f'{stock}_MA30' for stock in stocks]].head())


# In[78]:


# Create lag features for the news sentiment (1-day, 2-day, 3-day lags)
sentiment_columns = ['News Sentiment']  # Assuming 'News Sentiment' is the column for sentiment

for col in sentiment_columns:
    for lag in [1, 2, 3]:
        merged_df[f'{col}_lag{lag}'] = merged_df[col].shift(lag)

# Display the first few rows to verify the lagged sentiment features
print(merged_df[['date'] + [f'News Sentiment_lag{lag}' for lag in [1, 2, 3]]].head())


# In[ ]:





# In[95]:


# Compute 7-day and 30-day moving averages for news sentiment and its lagged versions
for col in ['News Sentiment'] + [f'News Sentiment_lag{lag}' for lag in [1, 2, 3]]:
    merged_df[f'{col}_MA7'] = merged_df[col].rolling(window=7).mean()
    merged_df[f'{col}_MA30'] = merged_df[col].rolling(window=30).mean()

# Now, print the first few rows of the data for verification
# Corrected print statement: ensure correct bracket matching
print(merged_df[['date'] + [f'News Sentiment_MA7'] + [f'News Sentiment_lag{lag}_MA7' for lag in [1, 2, 3]] + 
                [f'News Sentiment_MA30'] + [f'News Sentiment_lag{lag}_MA30' for lag in [1, 2, 3]]].head())


# In[96]:


# Check if merged_df exists in the environment
try:
    print(merged_df.head(20))
except NameError:
    print("merged_df is not defined. Please load the data again.")


# In[98]:


# Create interaction features between sentiment and its moving averages
for col in ['News Sentiment'] + [f'News Sentiment_lag{lag}' for lag in [1, 2, 3]]:
    merged_df[f'{col}_MA7_interaction'] = merged_df[col] * merged_df[f'{col}_MA7']
    merged_df[f'{col}_MA30_interaction'] = merged_df[col] * merged_df[f'{col}_MA30']

# Display the first few rows to verify the interaction features
print(merged_df[['date'] + [f'{col}_MA7_interaction' for col in ['News Sentiment'] + [f'News Sentiment_lag{lag}' for lag in [1, 2, 3]]]].head())
print(merged_df[['date'] + [f'{col}_MA30_interaction' for col in ['News Sentiment'] + [f'News Sentiment_lag{lag}' for lag in [1, 2, 3]]]].head())


# In[99]:


merged_df.ffill(inplace=True)


# In[100]:


# List of stock columns
stock_columns = ['INTC', 'ASML', 'AMAT', 'AMD', 'QCOM', 'TSM', 'TXN', 'AVGO', 'NVDA']

# Multiply stock prices by sentiment and lagged sentiment
for stock in stock_columns:
    merged_df[f'{stock}_Sentiment'] = merged_df[stock] * merged_df['News Sentiment']
    for lag in [1, 2, 3]:
        merged_df[f'{stock}_Sentiment_lag{lag}'] = merged_df[stock] * merged_df[f'News Sentiment_lag{lag}']

# Display the updated dataframe
print(merged_df.head())



# In[101]:


# Calculate price trends based on 7-day and 30-day moving averages
for stock in stock_columns:
    merged_df[f'{stock}_Trend_MA7'] = merged_df[stock].rolling(window=7).mean()
    merged_df[f'{stock}_Trend_MA30'] = merged_df[stock].rolling(window=30).mean()

# Display the updated dataframe
print(merged_df.head())


# In[102]:


# Calculate sentiment trends based on 7-day and 30-day moving averages
merged_df['Sentiment_Trend_MA7'] = merged_df['News Sentiment'].rolling(window=7).mean()
merged_df['Sentiment_Trend_MA30'] = merged_df['News Sentiment'].rolling(window=30).mean()

# Display the updated dataframe
print(merged_df.head())


# In[103]:


# Display the first 20 rows of the merged_df
print(merged_df.head(20))


# In[121]:


from sklearn.model_selection import train_test_split

X = merged_df.drop(columns=['date', 'target_column'])  # Replace 'target_column' with your actual target column
y = merged_df['target_column']  # Replace with the column you want to predict
y = merged_df['target_column']  # Replace with the column you want to predict

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[113]:


# Feature columns (including engineered features)
features = ['INTC', 'ASML', 'AMAT', 'AMD', 'QCOM', 'TSM', 'TXN', 'AVGO', 'NVDA',
            'News Sentiment', 'News Sentiment_lag1', 'News Sentiment_lag2', 
            'News Sentiment_lag3', 'News Sentiment_MA7', 'News Sentiment_MA30',
            'News Sentiment_lag1_MA7', 'News Sentiment_lag2_MA7', 'News Sentiment_lag3_MA7', 
            'News Sentiment_lag1_MA30', 'News Sentiment_lag2_MA30', 'News Sentiment_lag3_MA30', 
            'News Sentiment_MA7_interaction', 'News Sentiment_lag1_MA7_interaction',
            'News Sentiment_lag2_MA7_interaction', 'News Sentiment_lag3_MA7_interaction',
            'News Sentiment_MA30_interaction', 'News Sentiment_lag1_MA30_interaction', 
            'News Sentiment_lag2_MA30_interaction', 'News Sentiment_lag3_MA30_interaction']

# Assuming 'INTC' or any stock as a target variable (for example, 'INTC' price change)
merged_df['INTC_change'] = merged_df['INTC'].pct_change()  # Calculate percentage change in price

# Drop the first row with NaN value due to percentage change calculation
merged_df.dropna(subset=['INTC_change'], inplace=True)

# Features and target
X = merged_df[features]
y = merged_df['INTC_change']  # This is the target variable


# In[117]:


from sklearn.model_selection import train_test_split

X = merged_df.drop(columns=['date', 'target_column'])  # Replace 'target_column' with your actual target column
y = merged_df['target_column']  # Replace with the column you want to predict

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[115]:


from sklearn.ensemble import RandomForestRegressor

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
model.fit(X_train, y_train)



# In[116]:


from sklearn.metrics import mean_squared_error, r2_score

# Predict on test data
y_pred = model.predict(X_test)

# Calculate R² and MSE
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R²: {r2}")
print(f"Mean Squared Error: {mse}")


# In[ ]:





# In[ ]:




