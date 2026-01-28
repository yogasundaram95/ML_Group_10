#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GAN Models for Semiconductor Stock Prediction
1. TimeGAN - Synthetic data generation for augmentation
2. Stock-GAN - Adversarial prediction network
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)

# =============================================================================
# TIMEGAN - Synthetic Data Generation
# =============================================================================
class TimeGAN:
    """
    TimeGAN for generating synthetic time series data
    Based on: "Time-series Generative Adversarial Networks" (NeurIPS 2019)
    """

    def __init__(self, seq_len=24, n_features=10, hidden_dim=24, gamma=1):
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.gamma = gamma

        # Build networks
        self.embedder = self._build_embedder()
        self.recovery = self._build_recovery()
        self.generator = self._build_generator()
        self.supervisor = self._build_supervisor()
        self.discriminator = self._build_discriminator()

        # Optimizers
        self.autoencoder_opt = Adam(learning_rate=0.001)
        self.supervisor_opt = Adam(learning_rate=0.001)
        self.generator_opt = Adam(learning_rate=0.001)
        self.discriminator_opt = Adam(learning_rate=0.001)

    def _build_embedder(self):
        """Embedding network: maps real data to latent space"""
        inputs = layers.Input(shape=(self.seq_len, self.n_features))
        x = layers.GRU(self.hidden_dim, return_sequences=True)(inputs)
        x = layers.GRU(self.hidden_dim, return_sequences=True)(x)
        outputs = layers.Dense(self.hidden_dim, activation='sigmoid')(x)
        return Model(inputs, outputs, name='embedder')

    def _build_recovery(self):
        """Recovery network: maps latent space back to data space"""
        inputs = layers.Input(shape=(self.seq_len, self.hidden_dim))
        x = layers.GRU(self.hidden_dim, return_sequences=True)(inputs)
        x = layers.GRU(self.hidden_dim, return_sequences=True)(x)
        outputs = layers.Dense(self.n_features, activation='sigmoid')(x)
        return Model(inputs, outputs, name='recovery')

    def _build_generator(self):
        """Generator: creates synthetic latent representations"""
        inputs = layers.Input(shape=(self.seq_len, self.n_features))
        x = layers.GRU(self.hidden_dim, return_sequences=True)(inputs)
        x = layers.GRU(self.hidden_dim, return_sequences=True)(x)
        outputs = layers.Dense(self.hidden_dim, activation='sigmoid')(x)
        return Model(inputs, outputs, name='generator')

    def _build_supervisor(self):
        """Supervisor: captures temporal dynamics"""
        inputs = layers.Input(shape=(self.seq_len, self.hidden_dim))
        x = layers.GRU(self.hidden_dim, return_sequences=True)(inputs)
        outputs = layers.Dense(self.hidden_dim, activation='sigmoid')(x)
        return Model(inputs, outputs, name='supervisor')

    def _build_discriminator(self):
        """Discriminator: distinguishes real from synthetic"""
        inputs = layers.Input(shape=(self.seq_len, self.hidden_dim))
        x = layers.GRU(self.hidden_dim, return_sequences=True)(inputs)
        x = layers.GRU(self.hidden_dim, return_sequences=True)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        return Model(inputs, outputs, name='discriminator')

    def _get_random_batch(self, data, batch_size):
        """Get random batch of sequences"""
        idx = np.random.randint(0, len(data), batch_size)
        return data[idx]

    def _generate_noise(self, batch_size):
        """Generate random noise for generator input"""
        return np.random.uniform(0, 1, (batch_size, self.seq_len, self.n_features))

    @tf.function
    def _train_autoencoder_step(self, X):
        """Train embedding and recovery networks"""
        with tf.GradientTape() as tape:
            H = self.embedder(X, training=True)
            X_tilde = self.recovery(H, training=True)
            loss = tf.reduce_mean(tf.keras.losses.mse(X, X_tilde))

        vars = self.embedder.trainable_variables + self.recovery.trainable_variables
        grads = tape.gradient(loss, vars)
        self.autoencoder_opt.apply_gradients(zip(grads, vars))
        return loss

    @tf.function
    def _train_supervisor_step(self, X):
        """Train supervisor network"""
        with tf.GradientTape() as tape:
            H = self.embedder(X, training=False)
            H_hat_supervise = self.supervisor(H, training=True)
            loss = tf.reduce_mean(tf.keras.losses.mse(H[:, 1:, :], H_hat_supervise[:, :-1, :]))

        grads = tape.gradient(loss, self.supervisor.trainable_variables)
        self.supervisor_opt.apply_gradients(zip(grads, self.supervisor.trainable_variables))
        return loss

    @tf.function
    def _train_generator_step(self, X, Z):
        """Train generator network"""
        with tf.GradientTape() as tape:
            # Real embedding
            H = self.embedder(X, training=False)

            # Synthetic embedding
            E_hat = self.generator(Z, training=True)
            H_hat = self.supervisor(E_hat, training=False)

            # Discriminator outputs
            Y_fake = self.discriminator(H_hat, training=False)
            Y_fake_e = self.discriminator(E_hat, training=False)

            # Generator losses
            G_loss_U = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.ones_like(Y_fake), Y_fake, from_logits=False))
            G_loss_U_e = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.ones_like(Y_fake_e), Y_fake_e, from_logits=False))

            # Supervised loss
            G_loss_S = tf.reduce_mean(tf.keras.losses.mse(
                H[:, 1:, :], self.supervisor(self.generator(Z, training=True), training=False)[:, :-1, :]))

            # Moments loss
            G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X, [0])[1] + 1e-6) -
                                              tf.sqrt(tf.nn.moments(self.recovery(self.supervisor(
                                                  self.generator(Z, training=True), training=False),
                                                  training=False), [0])[1] + 1e-6)))
            G_loss_V2 = tf.reduce_mean(tf.abs(tf.nn.moments(X, [0])[0] -
                                              tf.nn.moments(self.recovery(self.supervisor(
                                                  self.generator(Z, training=True), training=False),
                                                  training=False), [0])[0]))
            G_loss_V = G_loss_V1 + G_loss_V2

            # Total generator loss
            G_loss = G_loss_U + self.gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V

        g_vars = self.generator.trainable_variables + self.supervisor.trainable_variables
        grads = tape.gradient(G_loss, g_vars)
        self.generator_opt.apply_gradients(zip(grads, g_vars))
        return G_loss

    @tf.function
    def _train_discriminator_step(self, X, Z):
        """Train discriminator network"""
        with tf.GradientTape() as tape:
            # Real embedding
            H = self.embedder(X, training=False)

            # Synthetic embedding
            E_hat = self.generator(Z, training=False)
            H_hat = self.supervisor(E_hat, training=False)

            # Discriminator outputs
            Y_real = self.discriminator(H, training=True)
            Y_fake = self.discriminator(H_hat, training=True)
            Y_fake_e = self.discriminator(E_hat, training=True)

            # Discriminator loss
            D_loss_real = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.ones_like(Y_real), Y_real, from_logits=False))
            D_loss_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.zeros_like(Y_fake), Y_fake, from_logits=False))
            D_loss_fake_e = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.zeros_like(Y_fake_e), Y_fake_e, from_logits=False))

            D_loss = D_loss_real + D_loss_fake + self.gamma * D_loss_fake_e

        grads = tape.gradient(D_loss, self.discriminator.trainable_variables)
        self.discriminator_opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return D_loss

    def fit(self, data, epochs=1000, batch_size=128, verbose=True):
        """
        Train TimeGAN

        Parameters:
        -----------
        data : np.array
            Shape (n_samples, seq_len, n_features)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        """
        print("\n=== Training TimeGAN ===")

        # Phase 1: Autoencoder training
        print("\nPhase 1: Training Autoencoder...")
        for epoch in range(epochs):
            X_batch = self._get_random_batch(data, batch_size)
            ae_loss = self._train_autoencoder_step(X_batch)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, AE Loss: {ae_loss:.4f}")

        # Phase 2: Supervisor training
        print("\nPhase 2: Training Supervisor...")
        for epoch in range(epochs):
            X_batch = self._get_random_batch(data, batch_size)
            s_loss = self._train_supervisor_step(X_batch)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Supervisor Loss: {s_loss:.4f}")

        # Phase 3: Joint training
        print("\nPhase 3: Joint Training (Generator + Discriminator)...")
        for epoch in range(epochs):
            # Generator training (2 steps)
            for _ in range(2):
                X_batch = self._get_random_batch(data, batch_size)
                Z_batch = self._generate_noise(batch_size)
                g_loss = self._train_generator_step(X_batch, Z_batch)

            # Discriminator training
            X_batch = self._get_random_batch(data, batch_size)
            Z_batch = self._generate_noise(batch_size)
            d_loss = self._train_discriminator_step(X_batch, Z_batch)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")

        print("\nTimeGAN training complete!")

    def generate(self, n_samples):
        """Generate synthetic data"""
        Z = self._generate_noise(n_samples)
        E_hat = self.generator(Z, training=False)
        H_hat = self.supervisor(E_hat, training=False)
        X_hat = self.recovery(H_hat, training=False)
        return X_hat.numpy()


# =============================================================================
# STOCK-GAN - Adversarial Prediction Network
# =============================================================================
class StockGAN:
    """
    GAN for stock price prediction
    Generator predicts future prices, Discriminator judges if predictions look realistic
    """

    def __init__(self, seq_len=30, n_features=10, pred_len=5, latent_dim=32):
        self.seq_len = seq_len
        self.n_features = n_features
        self.pred_len = pred_len
        self.latent_dim = latent_dim

        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

        self.g_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        self.d_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

        self.history = {'g_loss': [], 'd_loss': [], 'pred_loss': []}

    def _build_generator(self):
        """Generator: Takes past sequence + noise, outputs prediction"""
        # Historical data input
        hist_input = layers.Input(shape=(self.seq_len, self.n_features), name='hist_input')

        # Noise input for variability
        noise_input = layers.Input(shape=(self.latent_dim,), name='noise_input')

        # Process historical data
        x = layers.LSTM(64, return_sequences=True)(hist_input)
        x = layers.LSTM(32, return_sequences=False)(x)
        x = layers.BatchNormalization()(x)

        # Combine with noise
        noise_dense = layers.Dense(32, activation='relu')(noise_input)
        combined = layers.Concatenate()([x, noise_dense])

        # Generate prediction
        x = layers.Dense(64, activation='relu')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        # Output: predicted values for pred_len days
        output = layers.Dense(self.pred_len, activation='linear')(x)

        return Model([hist_input, noise_input], output, name='generator')

    def _build_discriminator(self):
        """Discriminator: Judges if sequence + prediction is realistic"""
        # Historical data input
        hist_input = layers.Input(shape=(self.seq_len, self.n_features), name='hist_input')

        # Prediction input
        pred_input = layers.Input(shape=(self.pred_len,), name='pred_input')

        # Process historical data
        x = layers.LSTM(64, return_sequences=True)(hist_input)
        x = layers.LSTM(32, return_sequences=False)(x)

        # Process prediction
        pred_dense = layers.Dense(32, activation='relu')(pred_input)

        # Combine
        combined = layers.Concatenate()([x, pred_dense])

        # Classify
        x = layers.Dense(64, activation='relu')(combined)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(1, activation='sigmoid')(x)

        return Model([hist_input, pred_input], output, name='discriminator')

    def _get_noise(self, batch_size):
        """Generate random noise"""
        return np.random.normal(0, 1, (batch_size, self.latent_dim))

    @tf.function
    def _train_step(self, X_hist, y_real):
        """Single training step"""
        batch_size = tf.shape(X_hist)[0]

        # Generate noise
        noise = tf.random.normal((batch_size, self.latent_dim))

        # Train Discriminator
        with tf.GradientTape() as d_tape:
            # Generate fake predictions
            y_fake = self.generator([X_hist, noise], training=False)

            # Discriminator outputs
            real_output = self.discriminator([X_hist, y_real], training=True)
            fake_output = self.discriminator([X_hist, y_fake], training=True)

            # Discriminator loss
            d_loss_real = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.ones_like(real_output) * 0.9, real_output))  # Label smoothing
            d_loss_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.zeros_like(fake_output), fake_output))
            d_loss = d_loss_real + d_loss_fake

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # Train Generator
        with tf.GradientTape() as g_tape:
            # Generate predictions
            y_fake = self.generator([X_hist, noise], training=True)

            # Discriminator output on fake
            fake_output = self.discriminator([X_hist, y_fake], training=False)

            # Generator adversarial loss (fool discriminator)
            g_loss_adv = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.ones_like(fake_output), fake_output))

            # Prediction loss (MSE with real values)
            g_loss_pred = tf.reduce_mean(tf.square(y_real - y_fake))

            # Total generator loss
            g_loss = g_loss_adv + 10.0 * g_loss_pred  # Weight prediction loss higher

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return g_loss, d_loss, g_loss_pred

    def fit(self, X_train, y_train, epochs=500, batch_size=32, verbose=True):
        """
        Train Stock-GAN

        Parameters:
        -----------
        X_train : np.array
            Historical sequences, shape (n_samples, seq_len, n_features)
        y_train : np.array
            Target values, shape (n_samples, pred_len)
        """
        print("\n=== Training Stock-GAN ===")

        n_samples = len(X_train)
        n_batches = n_samples // batch_size

        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            epoch_pred_loss = 0

            # Shuffle data
            idx = np.random.permutation(n_samples)
            X_shuffled = X_train[idx]
            y_shuffled = y_train[idx]

            for batch in range(n_batches):
                start = batch * batch_size
                end = start + batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                g_loss, d_loss, pred_loss = self._train_step(X_batch, y_batch)

                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                epoch_pred_loss += pred_loss

            # Average losses
            epoch_g_loss /= n_batches
            epoch_d_loss /= n_batches
            epoch_pred_loss /= n_batches

            self.history['g_loss'].append(float(epoch_g_loss))
            self.history['d_loss'].append(float(epoch_d_loss))
            self.history['pred_loss'].append(float(epoch_pred_loss))

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} - G Loss: {epoch_g_loss:.4f}, "
                      f"D Loss: {epoch_d_loss:.4f}, Pred Loss: {epoch_pred_loss:.6f}")

        print("\nStock-GAN training complete!")

    def predict(self, X, n_samples=1):
        """
        Generate predictions

        Parameters:
        -----------
        X : np.array
            Historical sequences
        n_samples : int
            Number of prediction samples (for uncertainty estimation)
        """
        predictions = []
        for _ in range(n_samples):
            noise = self._get_noise(len(X))
            pred = self.generator.predict([X, noise], verbose=0)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Return mean and std for uncertainty
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        return mean_pred, std_pred


# =============================================================================
# DATA PREPARATION UTILITIES
# =============================================================================
def prepare_sequences(data, seq_len, pred_len=1, target_col=None):
    """
    Prepare sequences for GAN training

    Parameters:
    -----------
    data : pd.DataFrame or np.array
        Input data
    seq_len : int
        Length of historical sequence
    pred_len : int
        Length of prediction horizon
    target_col : str
        Name of target column (if DataFrame)

    Returns:
    --------
    X : np.array
        Historical sequences (n_samples, seq_len, n_features)
    y : np.array
        Target values (n_samples, pred_len)
    """
    if isinstance(data, pd.DataFrame):
        if target_col:
            target_idx = data.columns.get_loc(target_col)
            values = data.values
        else:
            values = data.values
            target_idx = 0
    else:
        values = data
        target_idx = 0

    X, y = [], []

    for i in range(len(values) - seq_len - pred_len + 1):
        X.append(values[i:i+seq_len])
        if pred_len == 1:
            y.append(values[i+seq_len, target_idx])
        else:
            y.append(values[i+seq_len:i+seq_len+pred_len, target_idx])

    return np.array(X), np.array(y)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def run_gan_pipeline(data_path=None):
    """Run the complete GAN pipeline"""

    print("=" * 60)
    print("GAN Pipeline for Semiconductor Stock Prediction")
    print("=" * 60)

    # Load data
    if data_path is None:
        data_path = "Objective 1/data/enhanced_semiconductor_data.csv"

    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)

    # Select features for GAN
    feature_cols = ['Price_Change', 'ROC_5', 'RSI_14', 'MACD', 'Volatility_5',
                    'BB_PercentB', 'Stochastic_K', 'ATR_14', 'Momentum_10', 'Rolling_Vol_20']

    # Filter available columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    print(f"Using features: {feature_cols}")

    if len(feature_cols) < 3:
        print("Warning: Not enough features found. Using first 10 numeric columns.")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = numeric_cols[:10]

    # Prepare data
    data = df[feature_cols].copy()
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Scale data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Parameters
    SEQ_LEN = 30
    PRED_LEN = 5
    N_FEATURES = len(feature_cols)

    print(f"\nSequence length: {SEQ_LEN}")
    print(f"Prediction length: {PRED_LEN}")
    print(f"Number of features: {N_FEATURES}")

    # Prepare sequences
    X, y = prepare_sequences(data_scaled, SEQ_LEN, PRED_LEN, target_col=None)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    print(f"\nPrepared {len(X)} sequences")

    # Train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # =========================================================================
    # 1. TimeGAN - Data Augmentation
    # =========================================================================
    print("\n" + "=" * 60)
    print("1. TimeGAN - Synthetic Data Generation")
    print("=" * 60)

    timegan = TimeGAN(
        seq_len=SEQ_LEN,
        n_features=N_FEATURES,
        hidden_dim=24,
        gamma=1
    )

    # Train TimeGAN (reduced epochs for demo)
    timegan.fit(X_train, epochs=200, batch_size=64, verbose=True)

    # Generate synthetic data
    n_synthetic = len(X_train)  # Double the training data
    X_synthetic = timegan.generate(n_synthetic)

    print(f"\nGenerated {n_synthetic} synthetic sequences")
    print(f"Synthetic data shape: {X_synthetic.shape}")

    # Augment training data
    X_train_augmented = np.concatenate([X_train, X_synthetic], axis=0).astype(np.float32)
    y_train_augmented = np.concatenate([y_train, y_train], axis=0).astype(np.float32)  # Duplicate targets for synthetic

    print(f"Augmented training data: {len(X_train_augmented)} samples (3x original)")

    # =========================================================================
    # 2. Stock-GAN - Prediction
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. Stock-GAN - Adversarial Prediction")
    print("=" * 60)

    stockgan = StockGAN(
        seq_len=SEQ_LEN,
        n_features=N_FEATURES,
        pred_len=PRED_LEN,
        latent_dim=32
    )

    # Train on augmented data
    stockgan.fit(X_train_augmented, y_train_augmented, epochs=300, batch_size=32, verbose=True)

    # Make predictions
    y_pred_mean, y_pred_std = stockgan.predict(X_test, n_samples=10)

    # Evaluate
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # Use first prediction step for evaluation
    y_test_first = y_test[:, 0] if len(y_test.shape) > 1 else y_test
    y_pred_first = y_pred_mean[:, 0] if len(y_pred_mean.shape) > 1 else y_pred_mean

    mse = mean_squared_error(y_test_first, y_pred_first)
    mae = mean_absolute_error(y_test_first, y_pred_first)
    r2 = r2_score(y_test_first, y_pred_first)

    # Directional accuracy
    dir_true = np.sign(y_test_first)
    dir_pred = np.sign(y_pred_first)
    dir_acc = np.mean(dir_true == dir_pred)

    print("\n" + "=" * 60)
    print("STOCK-GAN RESULTS")
    print("=" * 60)
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.4f}")
    print(f"Directional Accuracy: {dir_acc:.2%}")

    # =========================================================================
    # 3. Visualizations
    # =========================================================================
    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Plot training history
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(stockgan.history['g_loss'], label='Generator')
    axes[0].set_title('Generator Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(stockgan.history['d_loss'], label='Discriminator')
    axes[1].set_title('Discriminator Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    axes[2].plot(stockgan.history['pred_loss'], label='Prediction', color='green')
    axes[2].set_title('Prediction Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/gan_training_history.png", dpi=150)
    print(f"Saved: {output_dir}/gan_training_history.png")
    plt.close()

    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_first, label='Actual', alpha=0.7)
    plt.plot(y_pred_first, label='GAN Prediction', alpha=0.7)
    plt.fill_between(range(len(y_pred_first)),
                     y_pred_first - y_pred_std[:, 0],
                     y_pred_first + y_pred_std[:, 0],
                     alpha=0.2, label='Uncertainty')
    plt.title('Stock-GAN Predictions vs Actual')
    plt.xlabel('Sample')
    plt.ylabel('Scaled Price Change')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gan_predictions.png", dpi=150)
    print(f"Saved: {output_dir}/gan_predictions.png")
    plt.close()

    # Save results
    results = {
        'Model': 'Stock-GAN',
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'Directional_Accuracy': dir_acc,
        'Training_Samples': len(X_train_augmented),
        'Synthetic_Samples': n_synthetic
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv(f"{output_dir}/gan_results.csv", index=False)
    print(f"Saved: {output_dir}/gan_results.csv")

    print("\n" + "=" * 60)
    print("GAN Pipeline Complete!")
    print("=" * 60)

    return timegan, stockgan, results


if __name__ == "__main__":
    timegan, stockgan, results = run_gan_pipeline()
