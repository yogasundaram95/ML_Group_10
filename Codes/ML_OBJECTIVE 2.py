#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np

# Load and sort by date
df = pd.read_csv(r"C:\Users\yoga sundaram\OneDrive - Cleveland State University\ML\stockSignal.csv")
df = df.sort_values('date')

# Calculate daily return (for NVDA)
df['nvda_return'] = df['NVDA'].pct_change()
df = df.dropna().reset_index(drop=True)


# In[7]:


# Get 10th and 90th percentiles
q_low = df['nvda_return'].quantile(0.10)
q_high = df['nvda_return'].quantile(0.90)

# Assign trend
def quantile_trend(x):
    if x <= q_low:
        return 'bearish'
    elif x >= q_high:
        return 'bullish'
    else:
        return 'neutral'

df['trend_quantile'] = df['nvda_return'].apply(quantile_trend)

# Check distribution
print(df['trend_quantile'].value_counts())


# In[8]:


mean = df['nvda_return'].mean()
std = df['nvda_return'].std()
df['zscore'] = (df['nvda_return'] - mean) / std

def zscore_trend(z):
    if z >= 1:
        return 'bullish'
    elif z <= -1:
        return 'bearish'
    else:
        return 'neutral'

df['trend_zscore'] = df['zscore'].apply(zscore_trend)

# Check distribution
print(df['trend_zscore'].value_counts())


# In[9]:


df['sma_10'] = df['NVDA'].rolling(window=10).mean()
df['sma_50'] = df['NVDA'].rolling(window=50).mean()

def crossover_trend(row):
    if row['sma_10'] > row['sma_50']:
        return 'bullish'
    elif row['sma_10'] < row['sma_50']:
        return 'bearish'
    else:
        return 'neutral'

df['trend_sma'] = df.apply(crossover_trend, axis=1)

# Drop NaNs introduced by SMA
df = df.dropna().reset_index(drop=True)

# Check distribution
print(df['trend_sma'].value_counts())


# In[10]:


print("Quantile-Based:\n", df['trend_quantile'].value_counts())
print("\nZ-Score-Based:\n", df['trend_zscore'].value_counts())
print("\nSMA Crossover:\n", df['trend_sma'].value_counts())


# In[12]:


from sklearn.preprocessing import LabelEncoder, StandardScaler

# Features: all stock prices (excluding date and trend columns)
feature_cols = ['INTC', 'ASML', 'AMAT', 'AMD', 'QCOM', 'TSM', 'TXN', 'AVGO', 'NVDA']
X = df[feature_cols]

# Target: use z-score-based trend
y = df['trend_zscore']

# Encode labels: 'bullish' → 2, 'neutral' → 1, 'bearish' → 0
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Check shapes
print(f"Features shape: {X_scaled.shape}")
print(f"Target shape: {y_encoded.shape}")
print(f"Label classes: {list(le.classes_)}")


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Initialize models
logreg = LogisticRegression(max_iter=1000)
svm = SVC(kernel='rbf', C=10, gamma=0.1, probability=True)
gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)

# Train models
logreg.fit(X_train, y_train)
svm.fit(X_train, y_train)
gbc.fit(X_train, y_train)

# Predict
models = {'Logistic Regression': logreg, 'SVM': svm, 'Gradient Boosting': gbc}

for name, model in models.items():
    print(f"\n{name}")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


# In[15]:


pip install imbalanced-learn


# In[16]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# First, re-split original X and y (in case of reruns)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", np.bincount(y_train))
print("After SMOTE:", np.bincount(y_train_smote))


# In[19]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Train Gradient Boosting Classifier on SMOTE data
gbc_smote = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
gbc_smote.fit(X_train_smote, y_train_smote)

# Predict on real-world (imbalanced) test set
y_pred_smote = gbc_smote.predict(X_test)

# Classification Report
print("📊 Gradient Boosting (After SMOTE Training)")
print(classification_report(y_test, y_pred_smote, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_smote)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Greens')
plt.title("GBC with SMOTE - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[2]:


pip install xgboost


# In[3]:


from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train_smote, y_train_smote)




# In[ ]:




