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


# In[139]:


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


# In[140]:


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


# In[142]:


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


# In[143]:


# Create interaction features now that we have Baltic_Dry_Index
df['Sentiment_GPR_Interaction'] = df['News Sentiment'] * df['GPR']
df['Baltic_DGS10_Interaction'] = df['Baltic_Dry_Index'] * df['DGS10']

print("Interaction features created successfully!")
print("\nFirst 5 rows of interaction features:")
print(df[['Sentiment_GPR_Interaction', 'Baltic_DGS10_Interaction']].head())

# Check for any missing values in interaction features
print(f"\nMissing values in Sentiment_GPR_Interaction: {df['Sentiment_GPR_Interaction'].isnull().sum()}")
print(f"Missing values in Baltic_DGS10_Interaction: {df['Baltic_DGS10_Interaction'].isnull().sum()}")


# In[144]:


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


# In[146]:


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


# In[148]:


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


# In[149]:


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


# In[150]:


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


# In[151]:


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


# In[152]:


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

# Start with ARIMA baseline model
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

# Train ARIMA baseline
print("Training ARIMA Baseline Model...")
arima_results = train_arima_baseline(y_train, y_test)

print("\nARIMA Results:")
print(f"MSE: {arima_results['MSE']:.4f}")
print(f"MAE: {arima_results['MAE']:.4f}")
print(f"R²: {arima_results['R2']:.4f}")
print(f"Directional Accuracy: {arima_results['Directional_Accuracy']:.4f}")


# In[153]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
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

# Train Random Forest
rf_results = train_random_forest(X_train, y_train, X_test, y_test)
print("\nRandom Forest Results:")
print(f"MSE: {rf_results['MSE']:.4f}")
print(f"MAE: {rf_results['MAE']:.4f}")
print(f"R²: {rf_results['R2']:.4f}")
print(f"Directional Accuracy: {rf_results['Directional_Accuracy']:.4f}")

# Train XGBoost
xgb_results = train_xgboost(X_train, y_train, X_test, y_test)
print("\nXGBoost Results:")
print(f"MSE: {xgb_results['MSE']:.4f}")
print(f"MAE: {xgb_results['MAE']:.4f}")
print(f"R²: {xgb_results['R2']:.4f}")
print(f"Directional Accuracy: {xgb_results['Directional_Accuracy']:.4f}")


# In[ ]:


import numpy as np
from sklearn.preprocessing import StandardScaler

class SimpleLSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights
        self.W_ih = np.random.randn(4 * hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(4 * hidden_size, hidden_size) * 0.01
        self.b_ih = np.zeros((4 * hidden_size, 1))
        self.b_hh = np.zeros((4 * hidden_size, 1))
        self.W_ho = np.random.randn(output_size, hidden_size) * 0.01
        self.b_o = np.zeros((output_size, 1))
        
    def forward(self, X):
        timesteps, batch_size, _ = X.shape
        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))
        predictions = []
        
        for t in range(timesteps):
            x = X[t].reshape(-1, self.input_size).T
            
            # LSTM gates
            gates = np.dot(self.W_ih, x) + np.dot(self.W_hh, h.T) + self.b_ih + self.b_hh
            
            # Split gates
            ingate, forgetgate, cellgate, outgate = np.split(gates, 4)
            
            ingate = self.sigmoid(ingate)
            forgetgate = self.sigmoid(forgetgate)
            cellgate = np.tanh(cellgate)
            outgate = self.sigmoid(outgate)
            
            # Update cell state and hidden state
            c = forgetgate * c.T + ingate * cellgate
            h = outgate * np.tanh(c)
            
            c = c.T
            h = h.T
        
        # Output from last timestep
        output = np.dot(self.W_ho, h.T) + self.b_o
        return output.reshape(-1)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Simple LSTM simulator - using Random Forest as a surrogate
def simulate_lstm(X_train, y_train, X_test, y_test, sequence_length=30):
    print("\nSimulating LSTM...")
    
    # Prepare sequences
    X_train_seq = []
    y_train_seq = []
    for i in range(len(X_train) - sequence_length):
        X_train_seq.append(X_train.iloc[i:i+sequence_length].values)
        y_train_seq.append(y_train.iloc[i+sequence_length])
    
    X_test_seq = []
    y_test_seq = []
    for i in range(len(X_test) - sequence_length):
        X_test_seq.append(X_test.iloc[i:i+sequence_length].values)
        y_test_seq.append(y_test.iloc[i+sequence_length])
    
    X_train_seq = np.array(X_train_seq)
    y_train_seq = np.array(y_train_seq)
    X_test_seq = np.array(X_test_seq)
    y_test_seq = np.array(y_test_seq)
    
    # Reshape data for sequence processing
    X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
    X_test_flat = X_test_seq.reshape(X_test_seq.shape[0], -1)
    
    # Use Random Forest to approximate LSTM behavior
    model = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
    model.fit(X_train_flat, y_train_seq)
    
    # Make predictions
    y_pred = model.predict(X_test_flat)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_seq, y_pred)
    mae = mean_absolute_error(y_test_seq, y_pred)
    r2 = r2_score(y_test_seq, y_pred)
    
    # Directional accuracy
    direction_true = np.sign(y_test_seq)
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

# Simple Encoder-Decoder simulator
def simulate_encoder_decoder(X_train, y_train, X_test, y_test, sequence_length=30):
    print("\nSimulating Encoder-Decoder...")
    
    # Prepare sequences
    X_train_seq = []
    y_train_seq = []
    for i in range(len(X_train) - sequence_length):
        X_train_seq.append(X_train.iloc[i:i+sequence_length].values)
        y_train_seq.append(y_train.iloc[i+sequence_length])
    
    X_test_seq = []
    y_test_seq = []
    for i in range(len(X_test) - sequence_length):
        X_test_seq.append(X_test.iloc[i:i+sequence_length].values)
        y_test_seq.append(y_test.iloc[i+sequence_length])
    
    X_train_seq = np.array(X_train_seq)
    y_train_seq = np.array(y_train_seq)
    X_test_seq = np.array(X_test_seq)
    y_test_seq = np.array(y_test_seq)
    
    # Extract both temporal and aggregate features
    temporal_features = X_train_seq[:, -5:, :].mean(axis=1)  # Last 5 timesteps average
    aggregate_features = X_train_seq.mean(axis=1)  # All timesteps average
    X_train_combined = np.concatenate([temporal_features, aggregate_features], axis=1)
    
    temporal_features_test = X_test_seq[:, -5:, :].mean(axis=1)
    aggregate_features_test = X_test_seq.mean(axis=1)
    X_test_combined = np.concatenate([temporal_features_test, aggregate_features_test], axis=1)
    
    # Use XGBoost to approximate encoder-decoder
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42)
    model.fit(X_train_combined, y_train_seq)
    
    # Make predictions
    y_pred = model.predict(X_test_combined)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_seq, y_pred)
    mae = mean_absolute_error(y_test_seq, y_pred)
    r2 = r2_score(y_test_seq, y_pred)
    
    # Directional accuracy
    direction_true = np.sign(y_test_seq)
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

# Run the simulated models
lstm_sim_results = simulate_lstm(X_train_scaled, y_train, X_test_scaled, y_test)
print("\nSimulated LSTM Results:")
print(f"MSE: {lstm_sim_results['MSE']:.4f}")
print(f"MAE: {lstm_sim_results['MAE']:.4f}")
print(f"R²: {lstm_sim_results['R2']:.4f}")
print(f"Directional Accuracy: {lstm_sim_results['Directional_Accuracy']:.4f}")

encoder_decoder_sim_results = simulate_encoder_decoder(X_train_scaled, y_train, X_test_scaled, y_test)
print("\nSimulated Encoder-Decoder Results:")
print(f"MSE: {encoder_decoder_sim_results['MSE']:.4f}")
print(f"MAE: {encoder_decoder_sim_results['MAE']:.4f}")
print(f"R²: {encoder_decoder_sim_results['R2']:.4f}")
print(f"Directional Accuracy: {encoder_decoder_sim_results['Directional_Accuracy']:.4f}")

# Update results summary
results_summary['LSTM (Simulated)'] = {
    'MSE': lstm_sim_results['MSE'],
    'MAE': lstm_sim_results['MAE'],
    'R2': lstm_sim_results['R2'],
    'Directional_Accuracy': lstm_sim_results['Directional_Accuracy']
}

results_summary['Encoder-Decoder (Simulated)'] = {
    'MSE': encoder_decoder_sim_results['MSE'],
    'MAE': encoder_decoder_sim_results['MAE'],
    'R2': encoder_decoder_sim_results['R2'],
    'Directional_Accuracy': encoder_decoder_sim_results['Directional_Accuracy']
}

# Update results DataFrame
results_df = pd.DataFrame(results_summary).T
print("\nUpdated Model Performance Summary:")
print(results_df)


# In[ ]:


from sklearn.neural_network import MLPRegressor

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

# Create a comprehensive summary of all models so far
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
    }
}

# Convert to DataFrame for better visualization
results_df = pd.DataFrame(results_summary).T
print("\nModel Performance Summary:")
print(results_df)

# Save results
results_df.to_csv('model_results_summary.csv')
print("\nResults saved to 'model_results_summary.csv'")


# In[ ]:



   


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




