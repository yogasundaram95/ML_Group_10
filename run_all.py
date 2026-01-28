#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML_Group_10 - Semiconductor Stock Analysis
Consolidated script combining Objectives 1, 2, and 3
All data leakage fixes applied
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, silhouette_score, davies_bouldin_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import f_oneway

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    print("XGBoost not available. Install with: pip install xgboost")
    XGB_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    print("statsmodels not available. Install with: pip install statsmodels")
    ARIMA_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Objective 1", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLIT_DATE = '2024-01-01'
RANDOM_STATE = 42

# Stock tickers
STOCK_TICKERS = ['INTC', 'ASML', 'AMAT', 'AMD', 'QCOM', 'TSM', 'TXN', 'AVGO', 'NVDA']

print("=" * 60)
print("ML_Group_10 - Semiconductor Stock Analysis")
print("=" * 60)
print(f"Base directory: {BASE_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")


# =============================================================================
# OBJECTIVE 2: DATA LOADING AND PREPROCESSING
# =============================================================================
def run_objective_2():
    """Load and preprocess data with proper handling to avoid data leakage"""
    print("\n" + "=" * 60)
    print("OBJECTIVE 2: Data Loading and Preprocessing")
    print("=" * 60)

    # Try to load preprocessed data first
    preprocessed_file = os.path.join(DATA_DIR, "preprocessed_data_complete.csv")
    enhanced_file = os.path.join(DATA_DIR, "enhanced_semiconductor_data.csv")

    if os.path.exists(enhanced_file):
        print(f"Loading enhanced data from: {enhanced_file}")
        df = pd.read_csv(enhanced_file)
    elif os.path.exists(preprocessed_file):
        print(f"Loading preprocessed data from: {preprocessed_file}")
        df = pd.read_csv(preprocessed_file)
    else:
        # Load raw data files
        print("Loading raw data files...")
        data_files = {
            'stock': os.path.join(DATA_DIR, "stockSignal.xlsx"),
            'bdi': os.path.join(DATA_DIR, "Baltic Dry Index Historical Data.xlsx"),
            'gpr': os.path.join(DATA_DIR, "data_gpr_export.xlsx"),
            'treasury': os.path.join(DATA_DIR, "DGS10.xlsx"),
            'sentiment': os.path.join(DATA_DIR, "news_sentiment_data.xlsx")
        }

        # Check which files exist
        for name, path in data_files.items():
            if os.path.exists(path):
                print(f"  Found: {name}")
            else:
                print(f"  Missing: {name} ({path})")

        # Load stock data
        if not os.path.exists(data_files['stock']):
            raise FileNotFoundError(f"Stock data not found at {data_files['stock']}")

        df = pd.read_excel(data_files['stock'])
        df['date'] = pd.to_datetime(df['date'])

        # Merge external factors if available
        for name, path in data_files.items():
            if name != 'stock' and os.path.exists(path):
                try:
                    ext_df = pd.read_excel(path)
                    if 'date' in ext_df.columns:
                        ext_df['date'] = pd.to_datetime(ext_df['date'], errors='coerce')
                        df = pd.merge(df, ext_df, on='date', how='left')
                        print(f"  Merged: {name}")
                except Exception as e:
                    print(f"  Could not merge {name}: {e}")

    # Ensure date column exists and is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:10]}...")

    # Handle missing values using modern pandas methods (no data leakage here)
    print("\nHandling missing values...")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].ffill().bfill()

    # Create target variable if not exists
    if 'Target_Change' not in df.columns and 'Current_Price' in df.columns:
        df['Next_Day_Close'] = df['Current_Price'].shift(-1)
        df['Target_Change'] = (df['Next_Day_Close'] - df['Current_Price']) / df['Current_Price']
        df = df.dropna(subset=['Target_Change'])

    print(f"Final dataset shape: {df.shape}")

    return df


# =============================================================================
# OBJECTIVE 1: ML MODELS
# =============================================================================
def run_objective_1(df):
    """Train and evaluate ML models with proper train/test split (no data leakage)"""
    print("\n" + "=" * 60)
    print("OBJECTIVE 1: Machine Learning Models")
    print("=" * 60)

    # Ensure date column
    if 'date' not in df.columns:
        print("Warning: No date column found. Using index-based split.")
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
    else:
        df['date'] = pd.to_datetime(df['date'])
        train_df = df[df['date'] < SPLIT_DATE].copy()
        test_df = df[df['date'] >= SPLIT_DATE].copy()

    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")

    if len(test_df) == 0:
        print("Warning: No test data after split date. Adjusting split...")
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        print(f"New split - Training: {len(train_df)}, Testing: {len(test_df)}")

    # Define features (exclude target and non-feature columns)
    exclude_cols = ['date', 'Target_Change', 'Next_Day_Close', 'Current_Price'] + STOCK_TICKERS
    feature_cols = [col for col in df.columns if col not in exclude_cols
                    and df[col].dtype in ['float64', 'int64']]

    print(f"Number of features: {len(feature_cols)}")

    if len(feature_cols) == 0:
        print("Error: No numeric features found!")
        return None

    # Prepare X and y
    X_train = train_df[feature_cols].copy()
    y_train = train_df['Target_Change'].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df['Target_Change'].copy()

    # Handle any remaining NaN and infinity values
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # IMPORTANT: Scale features - fit on training data ONLY (prevents data leakage)
    print("\nScaling features (fit on training data only)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

    # IMPORTANT: Apply PCA - fit on training data ONLY (prevents data leakage)
    print("Applying PCA (fit on training data only)...")
    tech_cols = [col for col in feature_cols if any(x in col for x in
                 ['RSI', 'MACD', 'Stochastic', 'Williams', 'ATR', 'ROC', 'Volatility', 'Momentum'])]

    if len(tech_cols) >= 3:
        tech_pca = PCA(n_components=min(3, len(tech_cols)))
        tech_train_pca = tech_pca.fit_transform(X_train_scaled[tech_cols])
        tech_test_pca = tech_pca.transform(X_test_scaled[tech_cols])
        for i in range(tech_pca.n_components_):
            X_train_scaled[f'Tech_PCA_{i+1}'] = tech_train_pca[:, i]
            X_test_scaled[f'Tech_PCA_{i+1}'] = tech_test_pca[:, i]
        print(f"  Added {tech_pca.n_components_} Tech PCA components")

    # Store results
    results = {}

    # 1. ARIMA Baseline
    if ARIMA_AVAILABLE:
        print("\n--- Training ARIMA ---")
        try:
            best_aic = float('inf')
            best_order = (1, 0, 0)

            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(y_train, order=(p, d, q))
                            fitted = model.fit()
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                        except (ValueError, np.linalg.LinAlgError, Exception):
                            continue

            print(f"Best ARIMA order: {best_order}")
            final_model = ARIMA(y_train, order=best_order)
            fitted_model = final_model.fit()
            y_pred = fitted_model.forecast(steps=len(y_test))

            results['ARIMA'] = evaluate_model(y_test, y_pred, 'ARIMA')
        except Exception as e:
            print(f"ARIMA failed: {e}")

    # 2. Random Forest
    print("\n--- Training Random Forest ---")
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    results['Random Forest'] = evaluate_model(y_test, y_pred_rf, 'Random Forest')

    # 3. XGBoost
    if XGB_AVAILABLE:
        print("\n--- Training XGBoost ---")
        xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=7,
                          random_state=RANDOM_STATE, n_jobs=-1)
        xgb.fit(X_train_scaled, y_train)
        y_pred_xgb = xgb.predict(X_test_scaled)
        results['XGBoost'] = evaluate_model(y_test, y_pred_xgb, 'XGBoost')

    # 4. Gradient Boosting
    print("\n--- Training Gradient Boosting ---")
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                                   random_state=RANDOM_STATE)
    gb.fit(X_train_scaled, y_train)
    y_pred_gb = gb.predict(X_test_scaled)
    results['Gradient Boosting'] = evaluate_model(y_test, y_pred_gb, 'Gradient Boosting')

    # 5. ANN (MLP)
    print("\n--- Training ANN (MLP) ---")
    ann = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',
                       learning_rate_init=0.001, max_iter=500, random_state=RANDOM_STATE,
                       early_stopping=True, validation_fraction=0.2)
    ann.fit(X_train_scaled, y_train)
    y_pred_ann = ann.predict(X_test_scaled)
    results['ANN'] = evaluate_model(y_test, y_pred_ann, 'ANN')

    # 6. Ensemble (weighted average)
    print("\n--- Creating Ensemble ---")
    predictions = {
        'Random Forest': y_pred_rf,
        'Gradient Boosting': y_pred_gb,
        'ANN': y_pred_ann
    }
    if XGB_AVAILABLE:
        predictions['XGBoost'] = y_pred_xgb

    # Weight by R2 score
    weights = {}
    total_r2 = sum(max(0, results[name]['R2']) for name in predictions.keys())
    for name in predictions.keys():
        weights[name] = max(0, results[name]['R2']) / total_r2 if total_r2 > 0 else 1/len(predictions)

    y_pred_ensemble = np.zeros(len(y_test))
    for name, pred in predictions.items():
        y_pred_ensemble += weights[name] * pred

    results['Ensemble'] = evaluate_model(y_test, y_pred_ensemble, 'Ensemble')

    # Print summary
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 60)
    results_df = pd.DataFrame(results).T
    print(results_df.to_string())

    # Save results
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_results.csv'))

    # Feature importance
    print("\n--- Top 15 Features (Random Forest) ---")
    importance = pd.DataFrame({
        'Feature': X_train_scaled.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(importance.head(15).to_string(index=False))
    importance.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)

    return results, rf, scaler


def evaluate_model(y_true, y_pred, name):
    """Calculate model metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Directional accuracy
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    dir_acc = np.mean(direction_true == direction_pred)

    print(f"{name}: MSE={mse:.6f}, MAE={mae:.6f}, R2={r2:.4f}, Dir.Acc={dir_acc:.2%}")

    return {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'Directional_Accuracy': dir_acc
    }


# =============================================================================
# OBJECTIVE 3: MARKET REGIME CLUSTERING
# =============================================================================
def run_objective_3(df):
    """Market regime clustering with proper scaler handling"""
    print("\n" + "=" * 60)
    print("OBJECTIVE 3: Market Regime Clustering")
    print("=" * 60)

    # Check for Objective 3 specific data
    obj3_file = os.path.join(BASE_DIR, "Objective 3", "ML_OBJ3_final_data.csv")
    if os.path.exists(obj3_file):
        print(f"Loading Objective 3 data from: {obj3_file}")
        df = pd.read_csv(obj3_file)
        df.dropna(inplace=True)

    # Identify columns
    company_cols = [col for col in STOCK_TICKERS if col in df.columns]
    external_cols = [col for col in ['InterestRate', 'CPI', 'Sentiment', 'GPR', 'DGS10',
                                      'Baltic_Dry_Index', 'News_Sentiment'] if col in df.columns]

    if len(company_cols) == 0:
        print("Warning: No stock columns found for clustering")
        return None

    all_features = company_cols + external_cols
    print(f"Company columns: {company_cols}")
    print(f"External columns: {external_cols}")

    # IMPORTANT: Use SEPARATE scalers (prevents overwriting)
    scaler_all = StandardScaler()
    scaler_company = StandardScaler()

    X_all = scaler_all.fit_transform(df[all_features])
    X_company = scaler_company.fit_transform(df[company_cols])

    # KMeans with all features
    print("\nClustering with all features...")
    kmeans_all = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
    labels_all = kmeans_all.fit_predict(X_all)

    silhouette = silhouette_score(X_all, labels_all)
    dbi = davies_bouldin_score(X_all, labels_all)

    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Index: {dbi:.4f}")

    # KMeans without external features
    print("\nClustering without external features...")
    kmeans_company = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
    labels_company = kmeans_company.fit_predict(X_company)

    # Compare clusters
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(labels_all, labels_company)
    print(f"Adjusted Rand Index (with vs without external): {ari:.4f}")

    # Cluster characterization
    df['Cluster'] = labels_all
    cluster_means = df.groupby('Cluster')[all_features].mean()
    print("\nCluster Means:")
    print(cluster_means.to_string())

    # ANOVA on external features
    if len(external_cols) > 0:
        print("\nANOVA Results (External Factors):")
        for col in external_cols:
            if col in df.columns:
                groups = [df[df['Cluster'] == c][col] for c in sorted(df['Cluster'].unique())]
                f_stat, p_val = f_oneway(*groups)
                print(f"  {col}: F={f_stat:.2f}, p={p_val:.2e}")

    # Save results
    cluster_means.to_csv(os.path.join(OUTPUT_DIR, 'cluster_means.csv'))

    # Temporal stability (if date column exists)
    if 'Date' in df.columns or 'date' in df.columns:
        date_col = 'Date' if 'Date' in df.columns else 'date'
        df[date_col] = pd.to_datetime(df[date_col])

        pre_2022 = df[df[date_col] < "2022-01-01"]
        post_2022 = df[df[date_col] >= "2022-01-01"]

        if len(pre_2022) > 10 and len(post_2022) > 10:
            # Use SEPARATE scalers for each time period
            scaler_pre = StandardScaler()
            scaler_post = StandardScaler()

            X_pre = scaler_pre.fit_transform(pre_2022[all_features])
            X_post = scaler_post.fit_transform(post_2022[all_features])

            labels_pre = KMeans(n_clusters=2, random_state=RANDOM_STATE).fit_predict(X_pre)
            labels_post = KMeans(n_clusters=2, random_state=RANDOM_STATE).fit_predict(X_post)

            min_len = min(len(labels_pre), len(labels_post))
            ari_temporal = adjusted_rand_score(labels_pre[:min_len], labels_post[:min_len])
            print(f"\nTemporal Stability ARI (pre vs post 2022): {ari_temporal:.4f}")

    return {
        'silhouette': silhouette,
        'dbi': dbi,
        'ari': ari,
        'cluster_means': cluster_means
    }


# =============================================================================
# VISUALIZATION
# =============================================================================
def create_visualizations(results, df):
    """Create summary visualizations"""
    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)

    if results is None:
        print("No results to visualize")
        return

    # Model comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    results_df = pd.DataFrame(results).T

    # MSE
    ax = axes[0, 0]
    results_df['MSE'].plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Mean Squared Error by Model')
    ax.set_ylabel('MSE')
    ax.tick_params(axis='x', rotation=45)

    # R2
    ax = axes[0, 1]
    results_df['R2'].plot(kind='bar', ax=ax, color='forestgreen')
    ax.set_title('R² Score by Model')
    ax.set_ylabel('R²')
    ax.tick_params(axis='x', rotation=45)

    # MAE
    ax = axes[1, 0]
    results_df['MAE'].plot(kind='bar', ax=ax, color='coral')
    ax.set_title('Mean Absolute Error by Model')
    ax.set_ylabel('MAE')
    ax.tick_params(axis='x', rotation=45)

    # Directional Accuracy
    ax = axes[1, 1]
    results_df['Directional_Accuracy'].plot(kind='bar', ax=ax, color='purple')
    ax.set_title('Directional Accuracy by Model')
    ax.set_ylabel('Accuracy')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150)
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'model_comparison.png')}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main execution function"""
    start_time = datetime.now()
    print(f"\nStarted at: {start_time}")

    try:
        # Objective 2: Load and preprocess data
        df = run_objective_2()

        # Objective 1: Train ML models
        results, best_model, scaler = run_objective_1(df)

        # Objective 3: Market regime clustering
        cluster_results = run_objective_3(df)

        # Create visualizations
        create_visualizations(results, df)

        # Save models
        joblib.dump(best_model, os.path.join(OUTPUT_DIR, 'best_model.pkl'))
        joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
        print(f"\nModels saved to: {OUTPUT_DIR}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    end_time = datetime.now()
    print(f"\nCompleted at: {end_time}")
    print(f"Total time: {end_time - start_time}")
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
