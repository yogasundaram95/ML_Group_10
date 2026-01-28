#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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


# In[3]:


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


# In[4]:


from sklearn.preprocessing import MinMaxScaler

# NOTE: MinMax scaling has been moved to AFTER train/test split to prevent data leakage
# The columns that need to be normalized are:
# columns_to_normalize = ['INTC', 'ASML', 'AMAT', 'AMD', 'QCOM', 'TSM', 'TXN', 'AVGO', 'NVDA', 'GPR', 'GPRT', 'GPRH', 'DGS10']
# Normalization will be applied properly during the train/test split phase (see StandardScaler section below)

# Display the data (unnormalized at this stage)
print(merged_df.head())


# In[5]:


# Forward-fill the missing values
merged_df.ffill(inplace=True)


# In[6]:


# Drop columns with too many missing values
merged_df.dropna(axis=1, thresh=merged_df.shape[0]*0.8, inplace=True)  # Keep columns with at least 80% non-null values


# In[7]:


# Save the final cleaned and normalized dataset
merged_df.to_csv('final_cleaned_merged_stock_data.csv', index=False)


# In[8]:


# Display the first 20 rows
display(merged_df.head(20))


# In[9]:


import pandas as pd
import numpy as np

# Load your merged data
df = pd.read_csv('final_cleaned_merged_stock_data.csv')

# Show basic info
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# Check first few rows
print("\nFirst 5 rows:")
print(df.head())


# In[10]:


# Let's load the Baltic Dry Index data again to see its structure
bdi_df = pd.read_excel('./Baltic Dry Index Historical Data.xlsx')
print("Baltic Dry Index columns:")
print(bdi_df.columns.tolist())
print("\nBaltic Dry Index first 5 rows:")
print(bdi_df.head())

# Check the date format in Baltic data
bdi_df['date'] = pd.to_datetime(bdi_df['date'], errors='coerce')
print("\nBaltic date range:")
print(f"From: {bdi_df['date'].min()} to {bdi_df['date'].max()}")


# In[11]:


# Rename Baltic columns to avoid conflicts
bdi_df_renamed = bdi_df.rename(columns={
    'Price': 'Baltic_Dry_Index',
    'Open': 'Baltic_Open',
    'High': 'Baltic_High',
    'Low': 'Baltic_Low',
    'Vol.': 'Baltic_Volume',
    'Change %': 'Baltic_Change'
})

print("Baltic columns after renaming:")
print(bdi_df_renamed.columns.tolist())

# Create Baltic dictionary for mapping
baltic_dict = dict(zip(bdi_df_renamed['date'], bdi_df_renamed['Baltic_Dry_Index']))

# Add Baltic_Dry_Index to the main dataframe
df['Baltic_Dry_Index'] = df['date'].map(baltic_dict)

# Check if it was added successfully
print("\nWas Baltic_Dry_Index added successfully?")
print('Baltic_Dry_Index' in df.columns)

# Check for missing values in Baltic_Dry_Index
print(f"\nMissing values in Baltic_Dry_Index: {df['Baltic_Dry_Index'].isnull().sum()}")

# Fill missing values using interpolation
df['Baltic_Dry_Index'] = df['Baltic_Dry_Index'].interpolate(method='linear')
print(f"Missing values after interpolation: {df['Baltic_Dry_Index'].isnull().sum()}")

# Show first 5 rows with Baltic data
print("\nFirst 5 rows with Baltic_Dry_Index:")
print(df[['date', 'Baltic_Dry_Index']].head())


# In[12]:


# Create interaction features now that we have Baltic_Dry_Index
df['Sentiment_GPR_Interaction'] = df['News Sentiment'] * df['GPR']
df['Baltic_DGS10_Interaction'] = df['Baltic_Dry_Index'] * df['DGS10']

print("Interaction features created successfully!")
print("\nFirst 5 rows of interaction features:")
print(df[['Sentiment_GPR_Interaction', 'Baltic_DGS10_Interaction']].head())

# Check for any missing values in interaction features
print(f"\nMissing values in Sentiment_GPR_Interaction: {df['Sentiment_GPR_Interaction'].isnull().sum()}")
print(f"Missing values in Baltic_DGS10_Interaction: {df['Baltic_DGS10_Interaction'].isnull().sum()}")


# In[13]:


# Fill missing values in GPR with forward fill method
df['GPR'] = df['GPR'].ffill()

# Recreate interaction features after filling missing values
df['Sentiment_GPR_Interaction'] = df['News Sentiment'] * df['GPR']
df['Baltic_DGS10_Interaction'] = df['Baltic_Dry_Index'] * df['DGS10']

print("Interaction features with filled GPR:")
print(f"Missing values in Sentiment_GPR_Interaction: {df['Sentiment_GPR_Interaction'].isnull().sum()}")
print(f"Missing values in Baltic_DGS10_Interaction: {df['Baltic_DGS10_Interaction'].isnull().sum()}")

# Harmonic Pattern Detection
def detect_harmonic_patterns(prices, high, low):
    """
    Detect harmonic patterns like Bat, Butterfly, Gartley, and Crab
    """
    patterns = pd.DataFrame(index=prices.index)
    
    # Swing highs and lows
    window = 5
    highs = high.rolling(window=window, center=True).max()
    lows = low.rolling(window=window, center=True).min()
    
    swing_highs = (high == highs) & (high.shift(1) < high) & (high.shift(-1) < high)
    swing_lows = (low == lows) & (low.shift(1) > low) & (low.shift(-1) > low)
    
    # Initialize pattern flags
    patterns['is_bat'] = 0
    patterns['is_butterfly'] = 0
    patterns['is_gartley'] = 0
    patterns['is_crab'] = 0
    
    # Find XABCD patterns
    for i in range(4*window, len(prices) - window):
        # Find last 5 swing points
        recent_highs = swing_highs[i-4*window:i].sum()
        recent_lows = swing_lows[i-4*window:i].sum()
        
        if recent_highs >= 3 and recent_lows >= 2:
            # Extract XABCD points
            try:
                x_idx = np.where(swing_highs[i-4*window:i])[0][-3] + i-4*window
                a_idx = np.where(swing_lows[i-4*window:i])[0][-2] + i-4*window
                b_idx = np.where(swing_highs[i-4*window:i])[0][-2] + i-4*window
                c_idx = np.where(swing_lows[i-4*window:i])[0][-1] + i-4*window
                d_idx = np.where(swing_highs[i-4*window:i])[0][-1] + i-4*window
                
                # Calculate ratios
                XA = high[x_idx] - low[a_idx]
                AB = high[b_idx] - low[a_idx]
                BC = high[b_idx] - low[c_idx]
                CD = high[d_idx] - low[c_idx]
                
                AB_XA = AB / XA if XA != 0 else 0
                BC_AB = BC / AB if AB != 0 else 0
                CD_BC = CD / BC if BC != 0 else 0
                AD_XA = (high[d_idx] - low[a_idx]) / XA if XA != 0 else 0
                
                # Check pattern ratios
                # Bat Pattern
                if 0.382 <= AB_XA <= 0.5 and 0.382 <= BC_AB <= 0.886 and 1.618 <= CD_BC <= 2.618 and 0.886 <= AD_XA <= 1:
                    patterns['is_bat'][i] = 1
                
                # Butterfly Pattern  
                if 0.786 <= AB_XA <= 1 and 0.382 <= BC_AB <= 0.886 and 1.618 <= CD_BC <= 2.618 and 1.27 <= AD_XA <= 1.618:
                    patterns['is_butterfly'][i] = 1
                
                # Gartley Pattern
                if 0.618 <= AB_XA <= 1 and 0.382 <= BC_AB <= 0.886 and 1.13 <= CD_BC <= 1.618 and 0.786 <= AD_XA <= 1:
                    patterns['is_gartley'][i] = 1
                    
                # Crab Pattern
                if 0.382 <= AB_XA <= 0.618 and 0.382 <= BC_AB <= 0.886 and 2.24 <= CD_BC <= 3.618 and 1.618 <= AD_XA <= 1.618:
                    patterns['is_crab'][i] = 1

            except (IndexError, ValueError, ZeroDivisionError):
                # Skip patterns that can't be calculated due to insufficient data or invalid values
                pass
    
    return patterns

# Apply harmonic pattern detection
harmonic_patterns = detect_harmonic_patterns(df['Price'], df['High'], df['Low'])

# Add harmonic patterns to main dataframe
df['harmonic_bat'] = harmonic_patterns['is_bat']
df['harmonic_butterfly'] = harmonic_patterns['is_butterfly']
df['harmonic_gartley'] = harmonic_patterns['is_gartley']
df['harmonic_crab'] = harmonic_patterns['is_crab']

# Add harmonic pattern score (sum of all patterns)
df['harmonic_pattern_score'] = df[['harmonic_bat', 'harmonic_butterfly', 'harmonic_gartley', 'harmonic_crab']].sum(axis=1)

print("\nHarmonic pattern features added!")
print(f"Bat patterns detected: {df['harmonic_bat'].sum()}")
print(f"Butterfly patterns detected: {df['harmonic_butterfly'].sum()}")
print(f"Gartley patterns detected: {df['harmonic_gartley'].sum()}")
print(f"Crab patterns detected: {df['harmonic_crab'].sum()}")

# Show first few rows with harmonic patterns
print("\nFirst few rows with harmonic patterns:")
print(df[['date', 'harmonic_pattern_score', 'harmonic_bat', 'harmonic_butterfly', 'harmonic_gartley', 'harmonic_crab']].head(20))


# In[14]:


# First, let's handle the remaining missing values in Sentiment_GPR_Interaction
df['Sentiment_GPR_Interaction'] = df['Sentiment_GPR_Interaction'].ffill()

# Check if all missing values are filled
print(f"Missing values after forward fill: {df['Sentiment_GPR_Interaction'].isnull().sum()}")

# If still have missing values, fill with backward fill
if df['Sentiment_GPR_Interaction'].isnull().sum() > 0:
    df['Sentiment_GPR_Interaction'] = df['Sentiment_GPR_Interaction'].bfill()

print(f"Missing values after backward fill: {df['Sentiment_GPR_Interaction'].isnull().sum()}")

# Simplified harmonic pattern detection
def detect_simplified_harmonic_patterns(prices, high, low, window=10):
    """
    Simplified harmonic pattern detection
    """
    patterns = pd.DataFrame(index=prices.index)
    
    # Rolling max and min to identify potential reversal points
    rolling_high = high.rolling(window=window, center=True).max()
    rolling_low = low.rolling(window=window, center=True).min()
    
    # Identify potential reversal points
    is_high_point = (high == rolling_high) & (high > high.shift(1)) & (high > high.shift(-1))
    is_low_point = (low == rolling_low) & (low < low.shift(1)) & (low < low.shift(-1))
    
    # Calculate price momentum and volatility
    returns = prices.pct_change()
    volatility = returns.rolling(window=window).std()
    rsi = calculate_rsi(prices, window)
    
    # Simple harmonic pattern score based on multiple factors
    patterns['harmonic_score'] = 0.0
    
    for i in range(window, len(prices) - window):
        score = 0
        
        # Check for significant price swings
        price_range = high[i-window:i+window].max() - low[i-window:i+window].min()
        recent_range = high[i-window//2:i+window//2].max() - low[i-window//2:i+window//2].min()
        
        if price_range != 0:
            range_ratio = recent_range / price_range
            
            # Add points for potential reversal conditions
            if is_high_point[i] and rsi[i] > 70:  # Overbought with high point
                score += 1
            if is_low_point[i] and rsi[i] < 30:   # Oversold with low point
                score += 1
            if 0.3 < range_ratio < 0.7:           # Fibonacci-like ratio
                score += 1
            if volatility[i] > volatility[i-window:i].mean() * 1.5:  # High volatility
                score += 1
        
        patterns['harmonic_score'].iloc[i] = score
    
    # Create harmonic pattern flag (score >= 2)
    patterns['harmonic_detected'] = (patterns['harmonic_score'] >= 2).astype(int)
    
    return patterns

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Apply simplified harmonic pattern detection
harmonic_patterns = detect_simplified_harmonic_patterns(df['Price'], df['High'], df['Low'])

# Add to main dataframe
df['harmonic_score'] = harmonic_patterns['harmonic_score']
df['harmonic_detected'] = harmonic_patterns['harmonic_detected']

print("\nSimplified harmonic patterns detected!")
print(f"Total patterns detected: {df['harmonic_detected'].sum()}")
print(f"Average harmonic score: {df['harmonic_score'].mean():.3f}")

# Show rows where patterns were detected
detected_patterns = df[df['harmonic_detected'] == 1][['date', 'Price', 'harmonic_score']]
print(f"\nFirst 10 detected patterns:")
print(detected_patterns.head(10))

# Add additional technical indicators for harmonic confirmation
df['rsi_14'] = calculate_rsi(df['Price'], 14)
df['price_to_sma_20'] = df['Price'] / df['Price'].rolling(20).mean()

print("\nAdded RSI and price-to-SMA-20 ratio indicators")


# In[15]:


# First, let's check if we have any volume-related columns
print("Checking for volume columns...")
volume_cols = [col for col in df.columns if 'Vol' in col or 'Volume' in col]
print("Volume-related columns found:", volume_cols)

# Add moving averages
df['MA_7'] = df['Price'].rolling(window=7).mean()
df['MA_30'] = df['Price'].rolling(window=30).mean()

# Add MACD
def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = calculate_macd(df['Price'])

# Add Bollinger Bands
df['BB_Middle'] = df['Price'].rolling(window=20).mean()
std = df['Price'].rolling(window=20).std()
df['BB_Upper'] = df['BB_Middle'] + (std * 2)
df['BB_Lower'] = df['BB_Middle'] - (std * 2)

# Skip volume indicators since Baltic_Volume doesn't exist
# Instead, add Baltic price-based indicators
df['Baltic_MA_7'] = df['Baltic_Dry_Index'].rolling(window=7).mean()
df['Baltic_MA_30'] = df['Baltic_Dry_Index'].rolling(window=30).mean()
df['Baltic_Price_Momentum'] = df['Baltic_Dry_Index'].pct_change(periods=5)

# Add volatility
df['Price_Volatility'] = df['Price'].pct_change().rolling(window=20).std()

# Add rate of change
df['ROC_10'] = df['Price'].pct_change(periods=10)

# Add stochastic oscillator
def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    return stoch_k, stoch_d

df['Stochastic_K'], df['Stochastic_D'] = calculate_stochastic(df['High'], df['Low'], df['Price'])

print("Technical indicators added successfully!")
print(f"\nTotal columns now: {len(df.columns)}")

# Show some technical indicators for verification
print("\nSample of technical indicators:")
indicator_cols = ['rsi_14', 'MA_7', 'MA_30', 'MACD', 'BB_Upper', 'BB_Lower', 'Price_Volatility', 'harmonic_score']
print(df[indicator_cols].head())


# In[16]:


# Create target variable (next day's price change)
df['Next_Day_Close'] = df['Price'].shift(-1)
df['Target_Change'] = (df['Next_Day_Close'] - df['Price']) / df['Price']

# Show target variable info
print("Target variable created!")
print(f"Missing values in target: {df['Target_Change'].isnull().sum()}")
print(f"Target distribution:")
print(df['Target_Change'].describe())

# Remove rows with missing target values (last row)
df_clean = df.dropna(subset=['Target_Change'])
print(f"\nRows after removing missing target: {len(df_clean)}")

# Handle any remaining missing values in other columns by forward filling
for col in df_clean.columns:
    if df_clean[col].isnull().sum() > 0:
        print(f"Filling {df_clean[col].isnull().sum()} missing values in {col}")
        df_clean[col] = df_clean[col].ffill()

        # If still have missing values, try backward fill
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].bfill()
        
        # If still have missing values, fill with 0
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(0)

# Verify no missing values remain
print("\nMissing values after cleaning:")
missing_cols = [col for col in df_clean.columns if df_clean[col].isnull().sum() > 0]
if missing_cols:
    print("Columns with missing values:")
    for col in missing_cols:
        print(f"{col}: {df_clean[col].isnull().sum()}")
else:
    print("No missing values!")

# Display final dataframe info
print(f"\nFinal dataframe shape: {df_clean.shape}")
print(f"Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")


# In[17]:


# Define features to use for modeling
feature_columns = [col for col in df_clean.columns if col not in ['date', 'Next_Day_Close', 'Target_Change']]

print(f"Total features available: {len(feature_columns)}")
print("\nFirst 10 feature columns:")
print(feature_columns[:10])

# Prepare X and y
X = df_clean[feature_columns]
y = df_clean['Target_Change']

# Split by date (80% train, 20% test)
split_date = '2024-01-01'
train_mask = df_clean['date'] < split_date
test_mask = df_clean['date'] >= split_date

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"\nTraining data: {X_train.shape}")
print(f"Testing data: {X_test.shape}")
print(f"Training date range: {df_clean[train_mask]['date'].min()} to {df_clean[train_mask]['date'].max()}")
print(f"Testing date range: {df_clean[test_mask]['date'].min()} to {df_clean[test_mask]['date'].max()}")

# Scale the features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames with feature names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("\nData scaled successfully!")


# In[18]:


# Save the processed data
df_clean.to_csv('final_preprocessed_data.csv', index=False)
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
X_train_scaled.to_csv('X_train_scaled.csv', index=False)
X_test_scaled.to_csv('X_test_scaled.csv', index=False)

# Save the scaler for later use
import joblib
joblib.dump(scaler, 'feature_scaler.pkl')

print("Data files saved successfully!")

# Create a summary of our preprocessing
summary = {
    'total_samples': len(df_clean),
    'total_features': len(feature_columns),
    'training_samples': len(X_train),
    'testing_samples': len(X_test),
    'date_range': f"{df_clean['date'].min()} to {df_clean['date'].max()}",
    'train_date_range': f"{df_clean[train_mask]['date'].min()} to {df_clean[train_mask]['date'].max()}",
    'test_date_range': f"{df_clean[test_mask]['date'].min()} to {df_clean[test_mask]['date'].max()}",
    'target_mean': y.mean(),
    'target_std': y.std(),
    'harmonic_patterns_detected': df_clean['harmonic_detected'].sum()
}

print("\nPreprocessing Summary:")
for key, value in summary.items():
    print(f"{key}: {value}")

# List of all features created
print("\nFeature Categories:")
tech_indicators = [col for col in feature_columns if any(x in col for x in ['MA_', 'RSI', 'MACD', 'BB_', 'Stochastic'])]
external_factors = [col for col in feature_columns if any(x in col for x in ['GPR', 'Baltic', 'DGS10', 'News'])]
harmonic_features = [col for col in feature_columns if 'harmonic' in col]
interaction_features = [col for col in feature_columns if '_Interaction' in col]

print(f"Technical indicators: {len(tech_indicators)}")
print(f"External factors: {len(external_factors)}")
print(f"Harmonic pattern features: {len(harmonic_features)}")
print(f"Interaction features: {len(interaction_features)}")
print(f"Other features: {len(feature_columns) - len(tech_indicators) - len(external_factors) - len(harmonic_features) - len(interaction_features)}")


# In[27]:


# Machine Learning Model
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ARIMA Model
def train_arima_baseline(y_train, y_test):
    """Train ARIMA baseline model using only historical prices"""
    # Find optimal parameters using AIC
    best_aic = float('inf')
    best_order = None
    
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    model = ARIMA(y_train, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except (ValueError, np.linalg.LinAlgError, Exception) as e:
                    # ARIMA fitting can fail for various numerical reasons
                    continue
    
    print(f"Best ARIMA order: {best_order}, AIC: {best_aic:.2f}")
    
    # Train final ARIMA model
    final_model = ARIMA(y_train, order=best_order)
    fitted_model = final_model.fit()
    
    # Make predictions
    y_pred = fitted_model.forecast(steps=len(y_test))
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Directional accuracy
    direction_true = np.sign(y_test)
    direction_pred = np.sign(y_pred)
    directional_accuracy = np.mean(direction_true == direction_pred)
    
    return {
        'model': fitted_model,
        'predictions': y_pred,
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy
    }

# Train ARIMA
arima_results = train_arima_baseline(y_train, y_test)

print("\nARIMA Results:")
print(f"MSE: {arima_results['MSE']:.4f}")
print(f"MAE: {arima_results['MAE']:.4f}")
print(f"R²: {arima_results['R2']:.4f}")
print(f"Directional Accuracy: {arima_results['Directional_Accuracy']:.4f}")


# In[28]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import time

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Random Forest Model
def train_random_forest(X_train, y_train, X_test, y_test):
    print("Training Random Forest...")
    start_time = time.time()
    
    # Define parameter grid for RandomizedSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(rf, param_grid, n_iter=20, cv=3, n_jobs=-1, verbose=1)
    random_search.fit(X_train, y_train)
    
    best_rf = random_search.best_estimator_
    print(f"Best parameters: {random_search.best_params_}")
    
    # Make predictions
    y_pred = best_rf.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Directional accuracy
    direction_true = np.sign(y_test)
    direction_pred = np.sign(y_pred)
    directional_accuracy = np.mean(direction_true == direction_pred)
    
    print(f"Training time: {time.time() - start_time:.2f} seconds")
    
    return {
        'model': best_rf,
        'predictions': y_pred,
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy
    }

# Train Random Forest
rf_results = train_random_forest(X_train, y_train, X_test, y_test)
print("\nRandom Forest Results:")
print(f"MSE: {rf_results['MSE']:.4f}")
print(f"MAE: {rf_results['MAE']:.4f}")
print(f"R²: {rf_results['R2']:.4f}")
print(f"Directional Accuracy: {rf_results['Directional_Accuracy']:.4f}")


# In[29]:


from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# XGBoost Model
def train_xgboost(X_train, y_train, X_test, y_test):
    print("\nTraining XGBoost...")
    start_time = time.time()
    
    # Define parameter distribution for RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    xgb = XGBRegressor(random_state=42)
    random_search = RandomizedSearchCV(xgb, param_dist, n_iter=20, cv=3, n_jobs=-1, verbose=1)
    random_search.fit(X_train, y_train)
    
    best_xgb = random_search.best_estimator_
    print(f"Best parameters: {random_search.best_params_}")
    
    # Make predictions
    y_pred = best_xgb.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Directional accuracy
    direction_true = np.sign(y_test)
    direction_pred = np.sign(y_pred)
    directional_accuracy = np.mean(direction_true == direction_pred)
    
    print(f"Training time: {time.time() - start_time:.2f} seconds")
    
    return {
        'model': best_xgb,
        'predictions': y_pred,
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy
    }

# Train XGBoost
xgb_results = train_xgboost(X_train, y_train, X_test, y_test)
print("\nXGBoost Results:")
print(f"MSE: {xgb_results['MSE']:.4f}")
print(f"MAE: {xgb_results['MAE']:.4f}")
print(f"R²: {xgb_results['R2']:.4f}")
print(f"Directional Accuracy: {xgb_results['Directional_Accuracy']:.4f}")


# In[30]:


import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ANN using scikit-learn
def train_ann_sklearn(X_train, y_train, X_test, y_test):
   print("\nTraining ANN (scikit-learn)...")
   
   # Create and train the neural network model
   ann_model = MLPRegressor(
       hidden_layer_sizes=(256, 128, 64, 32),
       activation='relu',
       learning_rate_init=0.001,
       max_iter=1000,
       random_state=42,
       early_stopping=True,
       validation_fraction=0.2
   )
   
   # Train the model
   ann_model.fit(X_train, y_train)
   
   # Make predictions
   y_pred = ann_model.predict(X_test)
   
   # Calculate metrics
   mse = mean_squared_error(y_test, y_pred)
   mae = mean_absolute_error(y_test, y_pred)
   r2 = r2_score(y_test, y_pred)
   
   # Directional accuracy
   direction_true = np.sign(y_test)
   direction_pred = np.sign(y_pred)
   directional_accuracy = np.mean(direction_true == direction_pred)
   
   return {
       'model': ann_model,
       'predictions': y_pred,
       'MSE': mse,
       'MAE': mae,
       'R2': r2,
       'Directional_Accuracy': directional_accuracy
   }

# Train ANN model
ann_results = train_ann_sklearn(X_train_scaled, y_train, X_test_scaled, y_test)
print("\nANN Results (scikit-learn):")
print(f"MSE: {ann_results['MSE']:.4f}")
print(f"MAE: {ann_results['MAE']:.4f}")
print(f"R²: {ann_results['R2']:.4f}")
print(f"Directional Accuracy: {ann_results['Directional_Accuracy']:.4f}")


# In[31]:


import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ridge Regression with polynomial features as LSTM alternative
def ridge_lstm_alternative(X_train, y_train, X_test, y_test):
   print("\nTraining Ridge Regression as LSTM alternative...")
   
   # Create polynomial features to capture non-linear patterns
   poly = PolynomialFeatures(degree=2, include_bias=False)
   X_train_poly = poly.fit_transform(X_train)
   X_test_poly = poly.transform(X_test)
   
   # Train Ridge model
   model = Ridge(alpha=1.0)
   model.fit(X_train_poly, y_train)
   
   # Make predictions
   y_pred = model.predict(X_test_poly)
   
   # Calculate metrics
   mse = mean_squared_error(y_test, y_pred)
   mae = mean_absolute_error(y_test, y_pred)
   r2 = r2_score(y_test, y_pred)
   
   # Directional accuracy
   direction_true = np.sign(y_test)
   direction_pred = np.sign(y_pred)
   directional_accuracy = np.mean(direction_true == direction_pred)
   
   return {
       'model': model,
       'predictions': y_pred,
       'MSE': mse,
       'MAE': mae,
       'R2': r2,
       'Directional_Accuracy': directional_accuracy
   }

# Gradient Boosting as Encoder-Decoder alternative
def gb_encoder_decoder_alternative(X_train, y_train, X_test, y_test):
   print("\nTraining Gradient Boosting as Encoder-Decoder alternative...")
   
   # Use Gradient Boosting to capture complex patterns
   model = GradientBoostingRegressor(
       n_estimators=100,
       learning_rate=0.1,
       max_depth=3,
       random_state=42
   )
   
   model.fit(X_train, y_train)
   
   # Make predictions
   y_pred = model.predict(X_test)
   
   # Calculate metrics
   mse = mean_squared_error(y_test, y_pred)
   mae = mean_absolute_error(y_test, y_pred)
   r2 = r2_score(y_test, y_pred)
   
   # Directional accuracy
   direction_true = np.sign(y_test)
   direction_pred = np.sign(y_pred)
   directional_accuracy = np.mean(direction_true == direction_pred)
   
   return {
       'model': model,
       'predictions': y_pred,
       'MSE': mse,
       'MAE': mae,
       'R2': r2,
       'Directional_Accuracy': directional_accuracy
   }

# Train the alternative models
ridge_results = ridge_lstm_alternative(X_train_scaled, y_train, X_test_scaled, y_test)
print("\nRidge Regression Results (LSTM alternative):")
print(f"MSE: {ridge_results['MSE']:.4f}")
print(f"MAE: {ridge_results['MAE']:.4f}")
print(f"R²: {ridge_results['R2']:.4f}")
print(f"Directional Accuracy: {ridge_results['Directional_Accuracy']:.4f}")

gb_results = gb_encoder_decoder_alternative(X_train_scaled, y_train, X_test_scaled, y_test)
print("\nGradient Boosting Results (Encoder-Decoder alternative):")
print(f"MSE: {gb_results['MSE']:.4f}")
print(f"MAE: {gb_results['MAE']:.4f}")
print(f"R²: {gb_results['R2']:.4f}")
print(f"Directional Accuracy: {gb_results['Directional_Accuracy']:.4f}")


# In[32]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create the complete results summary with all models
results_summary = {
   'ARIMA': {
       'MSE': arima_results['MSE'],
       'MAE': arima_results['MAE'],
       'R2': arima_results['R2'],
       'Directional_Accuracy': arima_results['Directional_Accuracy']
   },
   'Random Forest': {
       'MSE': rf_results['MSE'],
       'MAE': rf_results['MAE'],
       'R2': rf_results['R2'],
       'Directional_Accuracy': rf_results['Directional_Accuracy']
   },
   'XGBoost': {
       'MSE': xgb_results['MSE'],
       'MAE': xgb_results['MAE'],
       'R2': xgb_results['R2'],
       'Directional_Accuracy': xgb_results['Directional_Accuracy']
   },
   'ANN': {
       'MSE': ann_results['MSE'],
       'MAE': ann_results['MAE'],
       'R2': ann_results['R2'],
       'Directional_Accuracy': ann_results['Directional_Accuracy']
   },
   'Ridge (LSTM Alt)': {
       'MSE': ridge_results['MSE'],
       'MAE': ridge_results['MAE'],
       'R2': ridge_results['R2'],
       'Directional_Accuracy': ridge_results['Directional_Accuracy']
   },
   'Gradient Boosting (Enc-Dec Alt)': {
       'MSE': gb_results['MSE'],
       'MAE': gb_results['MAE'],
       'R2': gb_results['R2'],
       'Directional_Accuracy': gb_results['Directional_Accuracy']
   }
}

# Final results DataFrame
results_df = pd.DataFrame(results_summary).T
print("\nFinal Model Performance Summary:")
print(results_df)

# Create final visualization
plt.figure(figsize=(14, 10))
results_df.sort_values('MSE').plot(kind='bar')
plt.title('Comprehensive Model Performance Comparison')
plt.xlabel('Models')
plt.xticks(rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('final_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a comprehensive comparison plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# MSE comparison
results_df.sort_values('MSE')['MSE'].plot(kind='bar', ax=ax1, color='lightblue')
ax1.set_title('Mean Squared Error by Model')
ax1.set_ylabel('MSE')
ax1.tick_params(axis='x', rotation=45)

# R² comparison
results_df.sort_values('R2', ascending=False)['R2'].plot(kind='bar', ax=ax2, color='lightgreen')
ax2.set_title('R² Score by Model')
ax2.set_ylabel('R²')
ax2.tick_params(axis='x', rotation=45)

# Directional Accuracy comparison
results_df.sort_values('Directional_Accuracy', ascending=False)['Directional_Accuracy'].plot(kind='bar', ax=ax3, color='lightcoral')
ax3.set_title('Directional Accuracy by Model')
ax3.set_ylabel('Accuracy')
ax3.tick_params(axis='x', rotation=45)

# MAE comparison
results_df.sort_values('MAE')['MAE'].plot(kind='bar', ax=ax4, color='lightpink')
ax4.set_title('Mean Absolute Error by Model')
ax4.set_ylabel('MAE')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Answer research questions
print("\n" + "="*50)
print("RESEARCH QUESTIONS ANSWERS:")
print("="*50)

print("\n1. Which ML algorithm provides the highest accuracy?")
print(f"Best MSE: {results_df['MSE'].idxmin()} ({results_df['MSE'].min():.4f})")
print(f"Best MAE: {results_df['MAE'].idxmin()} ({results_df['MAE'].min():.4f})")
print(f"Best R²: {results_df['R2'].idxmax()} ({results_df['R2'].max():.4f})")
print(f"Best Directional Accuracy: {results_df['Directional_Accuracy'].idxmax()} ({results_df['Directional_Accuracy'].max():.4f})")

# Save final results
results_df.to_csv('final_model_results.csv')
print("\nResults saved to 'final_model_results.csv'")

# Feature importance visualization for Random Forest
rf_importance = pd.DataFrame({
   'Feature': X_train.columns,
   'Importance': rf_results['model'].feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(rf_importance['Feature'][:15][::-1], rf_importance['Importance'][:15][::-1])
plt.xlabel('Importance')
plt.title('Top 15 Features - Random Forest')
plt.tight_layout()
plt.savefig('rf_feature_importance.png', dpi=300)
plt.show()

# External factor impact analysis
external_factors = [col for col in X_train.columns if any(x in col for x in ['Baltic', 'GPR', 'DGS10', 'News'])]
technical_indicators = [col for col in X_train.columns if any(x in col for x in ['MA_', 'RSI', 'MACD', 'BB_'])]

print("\n2. External Factor Impact:")
print(f"Number of external factors: {len(external_factors)}")
print(f"Top 5 external factors (by Random Forest importance):")
ext_factors_importance = rf_importance[rf_importance['Feature'].isin(external_factors)].head()
print(ext_factors_importance)


# In[34]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 1. Comprehensive Performance Visualization
fig = plt.figure(figsize=(20, 15))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.2)

# Individual metric plots
ax1 = fig.add_subplot(gs[0, 0])
mse_data = results_df['MSE'].sort_values()
colors = ['green' if val < 0.04 else 'orange' if val < 0.1 else 'red' for val in mse_data]
bars1 = ax1.bar(mse_data.index, mse_data.values, color=colors)
ax1.set_title('Mean Squared Error by Model', fontsize=14, fontweight='bold')
ax1.set_ylabel('MSE', fontsize=12)
ax1.set_xticklabels(mse_data.index, rotation=45, ha='right')
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', ha='center', va='bottom', fontsize=10)

ax2 = fig.add_subplot(gs[0, 1])
r2_data = results_df['R2'].sort_values(ascending=False)
colors = ['green' if val > 0.05 else 'orange' if val > 0 else 'red' for val in r2_data]
bars2 = ax2.bar(r2_data.index, r2_data.values, color=colors)
ax2.set_title('R² Score by Model', fontsize=14, fontweight='bold')
ax2.set_ylabel('R²', fontsize=12)
ax2.set_xticklabels(r2_data.index, rotation=45, ha='right')
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=10)

ax3 = fig.add_subplot(gs[1, 0])
acc_data = results_df['Directional_Accuracy'].sort_values(ascending=False)
colors = ['green' if val > 0.6 else 'orange' if val > 0.5 else 'red' for val in acc_data]
bars3 = ax3.bar(acc_data.index, acc_data.values, color=colors)
ax3.set_title('Directional Accuracy by Model', fontsize=14, fontweight='bold')
ax3.set_ylabel('Accuracy', fontsize=12)
ax3.set_xticklabels(acc_data.index, rotation=45, ha='right')
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2%}', ha='center', va='bottom', fontsize=10)

ax4 = fig.add_subplot(gs[1, 1])
mae_data = results_df['MAE'].sort_values()
colors = ['green' if val < 0.14 else 'orange' if val < 0.2 else 'red' for val in mae_data]
bars4 = ax4.bar(mae_data.index, mae_data.values, color=colors)
ax4.set_title('Mean Absolute Error by Model', fontsize=14, fontweight='bold')
ax4.set_ylabel('MAE', fontsize=12)
ax4.set_xticklabels(mae_data.index, rotation=45, ha='right')
for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', ha='center', va='bottom', fontsize=10)

# Performance heatmap
ax5 = fig.add_subplot(gs[2, :])
heatmap_data = results_df.copy()
# Normalize the data for heatmap (lower is better for MSE/MAE, higher is better for R2/Accuracy)
heatmap_data['MSE'] = 1 - (heatmap_data['MSE'] - heatmap_data['MSE'].min()) / (heatmap_data['MSE'].max() - heatmap_data['MSE'].min())
heatmap_data['MAE'] = 1 - (heatmap_data['MAE'] - heatmap_data['MAE'].min()) / (heatmap_data['MAE'].max() - heatmap_data['MAE'].min())
heatmap_data['R2'] = (heatmap_data['R2'] - heatmap_data['R2'].min()) / (heatmap_data['R2'].max() - heatmap_data['R2'].min())
heatmap_data['Directional_Accuracy'] = (heatmap_data['Directional_Accuracy'] - heatmap_data['Directional_Accuracy'].min()) / (heatmap_data['Directional_Accuracy'].max() - heatmap_data['Directional_Accuracy'].min())

sns.heatmap(heatmap_data.T, annot=True, fmt='.2f', cmap='RdYlGn', cbar_kws={'label': 'Normalized Performance'}, ax=ax5)
ax5.set_title('Model Performance Heatmap (Normalized)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('comprehensive_model_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. External Factor Importance Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Top external factors
ext_factors_importance = rf_importance[rf_importance['Feature'].str.contains('Baltic|GPR|DGS10|News')].head(15)
bars = ax1.barh(ext_factors_importance['Feature'][::-1], ext_factors_importance['Importance'][::-1], color='skyblue')
ax1.set_xlabel('Importance Score', fontsize=12)
ax1.set_title('Top 15 External Factors (Random Forest)', fontsize=14, fontweight='bold')
for bar in bars:
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2,
             f'{width:.4f}', ha='left', va='center', fontsize=10)

# Category comparison
categories = {
    'Baltic Dry Index': [col for col in rf_importance['Feature'] if 'Baltic' in col and 'Interaction' not in col],
    'GPR Indicators': [col for col in rf_importance['Feature'] if 'GPR' in col and 'Interaction' not in col],
    'Economic': [col for col in rf_importance['Feature'] if 'DGS10' in col and 'Interaction' not in col],
    'News Sentiment': [col for col in rf_importance['Feature'] if 'News' in col and 'Interaction' not in col],
    'Interactions': [col for col in rf_importance['Feature'] if 'Interaction' in col]
}

category_importance = {}
for cat, features in categories.items():
    category_importance[cat] = rf_importance[rf_importance['Feature'].isin(features)]['Importance'].mean()

cat_df = pd.DataFrame(list(category_importance.items()), columns=['Category', 'Avg_Importance'])
cat_df = cat_df.sort_values('Avg_Importance', ascending=False)

bars2 = ax2.bar(cat_df['Category'], cat_df['Avg_Importance'], color=sns.color_palette("husl", len(cat_df)))
ax2.set_title('Average Importance by Factor Category', fontsize=14, fontweight='bold')
ax2.set_ylabel('Average Importance', fontsize=12)
ax2.set_xticklabels(cat_df['Category'], rotation=45, ha='right')
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('external_factor_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Prediction vs Actual Scatter Plots
fig, axes = plt.subplots(2, 3, figsize=(20, 15))
axes = axes.ravel()

# Use exact names from results_df
models = {
    'ARIMA': arima_results['predictions'],
    'Random Forest': rf_results['predictions'],
    'XGBoost': xgb_results['predictions'],
    'ANN': ann_results['predictions'],
    'Gradient Boosting (Enc-Dec Alt)': gb_results['predictions'],
    'Ridge (LSTM Alt)': ridge_results['predictions']
}

for idx, (model_name, predictions) in enumerate(models.items()):
    ax = axes[idx]
    
    # Align predictions with actual values
    y_true = y_test[:len(predictions)]
    
    # Create scatter plot
    scatter = ax.scatter(y_true, predictions, alpha=0.6, c=np.abs(y_true - predictions), 
                        cmap='RdYlBu_r', s=50)
    
    # Add perfect prediction line
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    
    # Add regression line
    z = np.polyfit(y_true, predictions, 1)
    p = np.poly1d(z)
    ax.plot(y_true, p(y_true), "r-", lw=2, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
    
    ax.set_xlabel('Actual Returns', fontsize=12)
    ax.set_ylabel('Predicted Returns', fontsize=12)
    # Use the exact model name from results_df
    ax.set_title(f'{model_name}\n(R²={results_df.loc[model_name, "R2"]:.3f})', fontsize=14, fontweight='bold')
    
    # Add correlation text
    correlation = np.corrcoef(y_true, predictions)[0,1]
    ax.text(0.05, 0.95, f'Corr: {correlation:.3f}', 
            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=11)
    
    ax.legend(loc='lower right')
    plt.colorbar(scatter, ax=ax, label='Prediction Error')

plt.tight_layout()
plt.savefig('prediction_scatter_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Create a summary table image
from matplotlib.table import Table

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

# Prepare data for table
summary_data = []
for model in results_df.index:
    summary_data.append([
        model,
        f"{results_df.loc[model, 'MSE']:.4f}",
        f"{results_df.loc[model, 'MAE']:.4f}",
        f"{results_df.loc[model, 'R2']:.4f}",
        f"{results_df.loc[model, 'Directional_Accuracy']:.2%}"
    ])

columns = ['Model', 'MSE', 'MAE', 'R²', 'Direction Accuracy']
table = ax.table(cellText=summary_data, colLabels=columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5)

# Style the header
for i in range(len(columns)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight best performers
for i, row in enumerate(summary_data):
    # Best MSE (Random Forest)
    if row[0] == results_df['MSE'].idxmin():
        table[(i+1, 1)].set_facecolor('#90EE90')  # Light green
    # Best R2 (Random Forest)
    if row[0] == results_df['R2'].idxmax():
        table[(i+1, 3)].set_facecolor('#90EE90')
    # Best Directional Accuracy (XGBoost)
    if row[0] == results_df['Directional_Accuracy'].idxmax():
        table[(i+1, 4)].set_facecolor('#90EE90')

plt.title('Model Performance Summary Table', fontsize=16, fontweight='bold', pad=20)
plt.savefig('performance_summary_table.png', dpi=300, bbox_inches='tight')
plt.show()

print("All visualizations have been created and saved!")


# In[39]:


# =============================================================================
# STRATEGY 1: Advanced Feature Engineering
# =============================================================================

# Add time-lagged features for more external factors
def create_advanced_features(df):
    # Create more sophisticated lagged features
    for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
        # Price-based features
        df[f'Price_lag_{lag}'] = df['Price'].shift(lag)
        df[f'Price_return_{lag}'] = df['Price'].pct_change(lag)
        
        # External factor lags
        df[f'Baltic_lag_{lag}'] = df['Baltic_Dry_Index'].shift(lag)
        df[f'GPR_lag_{lag}'] = df['GPR'].shift(lag)
        df[f'Sentiment_lag_{lag}'] = df['News Sentiment'].shift(lag)
        df[f'DGS10_lag_{lag}'] = df['DGS10'].shift(lag)
    
    # Rolling features (better for capturing trends)
    for window in [5, 10, 20, 50]:
        df[f'Price_mean_{window}'] = df['Price'].rolling(window).mean()
        df[f'Price_std_{window}'] = df['Price'].rolling(window).std()
        df[f'Price_momentum_{window}'] = df['Price'] - df['Price'].shift(window)
        df[f'Baltic_mean_{window}'] = df['Baltic_Dry_Index'].rolling(window).mean()
        df[f'Baltic_volatility_{window}'] = df['Baltic_Dry_Index'].pct_change().rolling(window).std()
        df[f'Sentiment_momentum_{window}'] = df['News Sentiment'] - df['News Sentiment'].shift(window)
    
    # Interaction features (more complex relationships)
    df['Baltic_Sentiment_interaction'] = df['Baltic_Dry_Index'] * df['News Sentiment']
    df['GPR_DGS10_interaction'] = df['GPR'] * df['DGS10']
    df['Sentiment_DGS10_interaction'] = df['News Sentiment'] * df['DGS10']
    df['Baltic_GPR_interaction'] = df['Baltic_Dry_Index'] * df['GPR']
    
    # Market regime indicators
    df['Price_trend'] = np.where(df['Price'] > df['Price'].rolling(20).mean(), 1, 0)
    df['Baltic_trend'] = np.where(df['Baltic_Dry_Index'] > df['Baltic_Dry_Index'].rolling(20).mean(), 1, 0)
    
    # Volatility clustering
    df['Price_volatility_regime'] = df['Price'].pct_change().rolling(20).std()
    df['High_volatility'] = (df['Price_volatility_regime'] > df['Price_volatility_regime'].quantile(0.75)).astype(int)
    
    return df

# =============================================================================
# STRATEGY 2: Ensemble Methods with Standard Libraries
# =============================================================================

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

def create_enhanced_ensemble(X_train, y_train, X_test, y_test):
    # Base models with diverse approaches
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=500, max_depth=20, 
                                             min_samples_split=2, random_state=42),
        'Extra Trees': ExtraTreesRegressor(n_estimators=500, max_depth=20, 
                                         min_samples_split=2, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.05, 
                               max_depth=7, subsample=0.9, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, 
                                                     max_depth=8, random_state=42),
        'Ridge': Ridge(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    }
    
    # Train all base models
    predictions = {}
    individual_results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        
        # Calculate individual performance
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        individual_results[name] = {'MSE': mse, 'R2': r2}
    
    # Create weighted ensemble
    weights = {}
    total_r2 = sum(res['R2'] for res in individual_results.values())
    
    for name, res in individual_results.items():
        if total_r2 > 0:
            weights[name] = max(0, res['R2']) / total_r2
        else:
            weights[name] = 1 / len(models)
    
    # Apply weights
    ensemble_pred = np.zeros(len(y_test))
    for name, weight in weights.items():
        ensemble_pred += weight * predictions[name]
    
    # Evaluate ensemble
    mse = mean_squared_error(y_test, ensemble_pred)
    mae = mean_absolute_error(y_test, ensemble_pred)
    r2 = r2_score(y_test, ensemble_pred)
    
    direction_true = np.sign(y_test)
    direction_pred = np.sign(ensemble_pred)
    directional_accuracy = np.mean(direction_true == direction_pred)
    
    return {
        'model': models,  # Dictionary of all models
        'predictions': ensemble_pred,
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy,
        'individual_results': individual_results,
        'weights': weights
    }

# =============================================================================
# STRATEGY 3: Advanced Gradient Boosting with Custom Parameters
# =============================================================================

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

def train_advanced_gradient_boosting(X_train, y_train, X_test, y_test):
    # Try different gradient boosting configurations
    models = {
        'HistGB': HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.01, max_depth=10, 
            early_stopping=True, validation_fraction=0.1, random_state=42
        ),
        'GB_Deep': GradientBoostingRegressor(
            n_estimators=1000, learning_rate=0.01, max_depth=10, 
            subsample=0.8, max_features='sqrt', random_state=42
        ),
        'GB_Wide': GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=15, 
            subsample=0.9, max_features=0.8, random_state=42
        )
    }
    
    best_model = None
    best_performance = float('inf')
    all_predictions = {}
    all_results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_predictions[name] = y_pred
        
        mse = mean_squared_error(y_test, y_pred)
        if mse < best_performance:
            best_performance = mse
            best_model = model
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        direction_true = np.sign(y_test)
        direction_pred = np.sign(y_pred)
        directional_accuracy = np.mean(direction_true == direction_pred)
        
        all_results[name] = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'Directional_Accuracy': directional_accuracy
        }
    
    # Return best model results
    best_name = min(all_results, key=lambda x: all_results[x]['MSE'])
    best_result = all_results[best_name]
    
    return {
        'model': best_model,
        'predictions': all_predictions[best_name],
        'MSE': best_result['MSE'],
        'MAE': best_result['MAE'],
        'R2': best_result['R2'],
        'Directional_Accuracy': best_result['Directional_Accuracy'],
        'all_results': all_results
    }

# =============================================================================
# STRATEGY 4: Feature Selection and Dimensionality Reduction
# =============================================================================

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA

def optimize_features(X_train, y_train, X_test, k_features=100):
    # Select top k features using ANOVA f-score
    selector = SelectKBest(score_func=f_regression, k=k_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    # Apply PCA for dimensionality reduction (optional)
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    X_train_pca = pca.fit_transform(X_train_selected)
    X_test_pca = pca.transform(X_test_selected)
    
    return X_train_selected, X_test_selected, X_train_pca, X_test_pca, selector, pca, selected_features

# =============================================================================
# STRATEGY 5: Rolling Window Training
# =============================================================================

def rolling_window_train(X_train, y_train, X_test, y_test, window_size=500):
    predictions = []
    
    for i in range(len(X_test)):
        # Determine training window
        start_idx = max(0, len(X_train) - window_size)
        
        # Combine training data with historical test data
        if i > 0:
            combined_X = pd.concat([X_train.iloc[start_idx:], X_test.iloc[:i]])
            combined_y = pd.concat([y_train.iloc[start_idx:], y_test.iloc[:i]])
        else:
            combined_X = X_train.iloc[start_idx:]
            combined_y = y_train.iloc[start_idx:]
        
        # Train model on window
        model = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
        model.fit(combined_X, combined_y)
        
        # Predict next point
        pred = model.predict(X_test.iloc[[i]])
        predictions.append(pred[0])
    
    y_pred = np.array(predictions)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    direction_true = np.sign(y_test)
    direction_pred = np.sign(y_pred)
    directional_accuracy = np.mean(direction_true == direction_pred)
    
    return {
        'model': 'Rolling Window',
        'predictions': y_pred,
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy
    }

# =============================================================================
# IMPLEMENTATION: Run Improved Strategies
# =============================================================================

print("Implementing Advanced Strategies...")

# Step 1: Feature Engineering
enhanced_df = create_advanced_features(df_clean.copy())
enhanced_df.dropna(inplace=True)

# Prepare data
features = [col for col in enhanced_df.columns if col not in ['date', 'Next_Day_Close', 'Target_Change']]
X = enhanced_df[features]
y = enhanced_df['Target_Change']

# Split data
split_date = '2024-01-01'
train_mask = enhanced_df['date'] < split_date
test_mask = enhanced_df['date'] >= split_date

X_train_enhanced, X_test_enhanced = X[train_mask], X[test_mask]
y_train_enhanced, y_test_enhanced = y[train_mask], y[test_mask]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enhanced)
X_test_scaled = scaler.transform(X_test_enhanced)

# Convert back to DataFrames
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_enhanced.columns, index=X_train_enhanced.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_enhanced.columns, index=X_test_enhanced.index)

# Step 2: Feature Selection
k_features = 100  # Adjust based on your needs
X_train_selected, X_test_selected, X_train_pca, X_test_pca, selector, pca, selected_features = optimize_features(
    X_train_scaled, y_train_enhanced, X_test_scaled, k_features
)

print(f"\nSelected {len(selected_features)} features")
print("Top 10 most important features:", selected_features[:10])

# Step 3: Train Improved Models
print("\nTraining Enhanced Ensemble...")
ensemble_results = create_enhanced_ensemble(X_train_selected, y_train_enhanced, X_test_selected, y_test_enhanced)

print("\nTraining Advanced Gradient Boosting...")
gb_results = train_advanced_gradient_boosting(X_train_selected, y_train_enhanced, X_test_selected, y_test_enhanced)

print("\nTraining Rolling Window Model...")
rolling_results = rolling_window_train(X_train_scaled, y_train_enhanced, X_test_scaled, y_test_enhanced)

# Compare Results
print("\n" + "="*50)
print("PERFORMANCE IMPROVEMENT RESULTS")
print("="*50)

results_comparison = {
    'Original Random Forest': rf_results,
    'Enhanced Ensemble': ensemble_results,
    'Advanced Gradient Boosting': gb_results,
    'Rolling Window': rolling_results
}

for model_name, results in results_comparison.items():
    print(f"\n{model_name}:")
    print(f"MSE: {results['MSE']:.4f}")
    print(f"MAE: {results['MAE']:.4f}")
    print(f"R²: {results['R2']:.4f}")
    print(f"Directional Accuracy: {results['Directional_Accuracy']:.4f}")

# Print ensemble weights
print("\nEnsemble Weights:")
for name, weight in ensemble_results['weights'].items():
    print(f"{name}: {weight:.4f}")

# Print individual model performance in ensemble
print("\nIndividual Model Performance in Ensemble:")
for name, res in ensemble_results['individual_results'].items():
    print(f"{name}: MSE={res['MSE']:.4f}, R²={res['R2']:.4f}")

# Save improved models
joblib.dump(ensemble_results['model'], 'enhanced_ensemble_models.pkl')
joblib.dump(gb_results['model'], 'enhanced_gradient_boosting.pkl')
joblib.dump(scaler, 'enhanced_scaler.pkl')
joblib.dump(selector, 'feature_selector.pkl')
joblib.dump(pca, 'pca_transformer.pkl')

# Create improved data file
enhanced_df.to_csv('enhanced_stock_data.csv', index=False)

# Save enhanced features list
with open('enhanced_features.txt', 'w') as f:
    f.write('\n'.join(selected_features))

print("\nAll models and data saved successfully!")
print(f"Total features in enhanced dataset: {len(enhanced_df.columns)}")
print(f"Selected features for modeling: {len(selected_features)}")


# In[45]:


# =============================================================================
# STRATEGY: Stacked Model with Meta-Learner
# =============================================================================

from sklearn.ensemble import StackingRegressor

def create_stacked_model(X_train, y_train, X_test, y_test):
    # Define base models
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42)),
        ('xgb', xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=8, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=500, learning_rate=0.02, max_depth=8, random_state=42)),
        ('extra', ExtraTreesRegressor(n_estimators=400, max_depth=30, random_state=42))
    ]
    
    # Define meta-learner
    meta_model = Ridge(alpha=0.1)
    
    # Create stacking model
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    )
    
    # Train
    stacking_model.fit(X_train, y_train)
    
    # Predict
    y_pred_stacked = stacking_model.predict(X_test)
    
    # Evaluate
    mse_stacked = mean_squared_error(y_test, y_pred_stacked)
    mae_stacked = mean_absolute_error(y_test, y_pred_stacked)
    r2_stacked = r2_score(y_test, y_pred_stacked)
    
    direction_true = np.sign(y_test)
    direction_pred = np.sign(y_pred_stacked)
    directional_accuracy_stacked = np.mean(direction_true == direction_pred)
    
    return {
        'model': stacking_model,
        'predictions': y_pred_stacked,
        'MSE': mse_stacked,
        'MAE': mae_stacked,
        'R2': r2_stacked,
        'Directional_Accuracy': directional_accuracy_stacked
    }

# First, let's check the current indices
print("Checking indices...")
print(f"y_train_enhanced index: {y_train_enhanced.index[:10]}")
print(f"X_train_selected is a {type(X_train_selected)}")

# Reset indices to make them compatible with KFold
y_train_reset = y_train_enhanced.reset_index(drop=True)
y_test_reset = y_test_enhanced.reset_index(drop=True)

# If X_train_selected and X_test_selected are numpy arrays, we need to convert them
# or if they're DataFrames, reset their indices too
if isinstance(X_train_selected, pd.DataFrame):
    X_train_reset = X_train_selected.reset_index(drop=True)
    X_test_reset = X_test_selected.reset_index(drop=True)
else:
    X_train_reset = X_train_selected
    X_test_reset = X_test_selected

print("\nReset indices completed. Training optimized ensemble...")

# Train the optimized ensemble with reset indices
optimized_ensemble = OptimizedEnsemble(n_folds=5)
optimized_ensemble.fit(X_train_reset, y_train_reset)

# Make predictions
y_pred_optimized = optimized_ensemble.predict(X_test_reset)

# Evaluate
mse_optimized = mean_squared_error(y_test_reset, y_pred_optimized)
mae_optimized = mean_absolute_error(y_test_reset, y_pred_optimized)
r2_optimized = r2_score(y_test_reset, y_pred_optimized)

direction_true = np.sign(y_test_reset)
direction_pred = np.sign(y_pred_optimized)
directional_accuracy_optimized = np.mean(direction_true == direction_pred)

optimized_results = {
    'model': optimized_ensemble,
    'predictions': y_pred_optimized,
    'MSE': mse_optimized,
    'MAE': mae_optimized,
    'R2': r2_optimized,
    'Directional_Accuracy': directional_accuracy_optimized
}

print("\nTraining Stacked Model...")
stacked_results = create_stacked_model(X_train_reset, y_train_reset, X_test_reset, y_test_reset)

# Continue with the rest of the analysis...
print("\n" + "="*50)
print("FINAL PERFORMANCE COMPARISON")
print("="*50)

all_results = {
    'Original Random Forest': rf_results,
    'Advanced Gradient Boosting': gb_results,
    'Optimized Ensemble': optimized_results,
    'Stacked Model': stacked_results
}

for model_name, results in all_results.items():
    print(f"\n{model_name}:")
    print(f"MSE: {results['MSE']:.4f}")
    print(f"MAE: {results['MAE']:.4f}")
    print(f"R²: {results['R2']:.4f}")
    print(f"Directional Accuracy: {results['Directional_Accuracy']:.4f}")

# Print optimized weights
print("\n" + "="*50)
print("OPTIMIZED ENSEMBLE WEIGHTS:")
print("="*50)
for name, weight in optimized_ensemble.weights.items():
    print(f"{name}: {weight:.4f}")

# Print average cross-validation scores
print("\n" + "="*50)
print("CROSS-VALIDATION MSE SCORES:")
print("="*50)
for name, scores in optimized_ensemble.cv_scores.items():
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"{name}: {avg_score:.4f} ± {std_score:.4f}")

# Create performance comparison visualization
import matplotlib.pyplot as plt

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Extract metrics for plotting
models = list(all_results.keys())
mse_values = [all_results[m]['MSE'] for m in models]
r2_values = [all_results[m]['R2'] for m in models]
mae_values = [all_results[m]['MAE'] for m in models]
accuracy_values = [all_results[m]['Directional_Accuracy'] for m in models]

# Create bar plots
x = range(len(models))
width = 0.6

ax1.bar(x, mse_values, width, color=['lightblue', 'lightcoral', 'lightgreen', 'plum'])
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.set_ylabel('MSE')
ax1.set_title('Mean Squared Error')

ax2.bar(x, r2_values, width, color=['lightblue', 'lightcoral', 'lightgreen', 'plum'])
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.set_ylabel('R²')
ax2.set_title('R² Score')

ax3.bar(x, mae_values, width, color=['lightblue', 'lightcoral', 'lightgreen', 'plum'])
ax3.set_xticks(x)
ax3.set_xticklabels(models, rotation=45, ha='right')
ax3.set_ylabel('MAE')
ax3.set_title('Mean Absolute Error')

ax4.bar(x, accuracy_values, width, color=['lightblue', 'lightcoral', 'lightgreen', 'plum'])
ax4.set_xticks(x)
ax4.set_xticklabels(models, rotation=45, ha='right')
ax4.set_ylabel('Accuracy')
ax4.set_title('Directional Accuracy')

plt.tight_layout()
plt.savefig('optimized_model_comparison.png', dpi=300)
plt.show()

print("\nPerformance visualization saved!")


# In[ ]:




