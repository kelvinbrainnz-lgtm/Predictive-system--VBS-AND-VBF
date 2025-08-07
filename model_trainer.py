import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ================================================
# PHYSICS-BASED DATA GENERATION
# ================================================
def generate_physics_data(num_events=50000):
    """Generate dataset with clear VBS/VBF differentiation"""
    # VBS events (30% of dataset)
    num_vbs = int(num_events * 0.3)
    num_vbf = num_events - num_vbs
    
    data = {}
    
    # VBS Characteristics (Vector Boson Scattering)
    data['m_jj'] = np.concatenate([
        np.abs(np.random.normal(800, 150, num_vbs)),       # High dijet mass
        np.abs(np.random.normal(500, 100, num_vbf))        # Lower for VBF
    ])
    
    data['delta_eta_jj'] = np.concatenate([
        np.abs(np.random.normal(5.0, 1.0, num_vbs)),      # Large pseudorapidity
        np.abs(np.random.normal(3.0, 0.8, num_vbf))       # Smaller for VBF
    ])
    
    # Polarization - critical differentiator
    data['f_L'] = np.concatenate([
        np.random.beta(9, 1.5, num_vbs),                  # High longitudinal for VBS
        np.random.beta(1.5, 9, num_vbf)                   # Low for VBF
    ])
    
    data['f_T'] = np.concatenate([
        np.random.beta(1.5, 9, num_vbs),                  # Low transverse for VBS
        np.random.beta(9, 1.5, num_vbf)                   # High for VBF
    ])
    
    # Other features
    data['pt_j1'] = np.concatenate([
        np.abs(np.random.normal(140, 30, num_vbs)),       # Higher pT jets for VBS
        np.abs(np.random.normal(100, 30, num_vbf))
    ])
    
    data['pt_j2'] = np.concatenate([
        np.abs(np.random.normal(90, 25, num_vbs)),
        np.abs(np.random.normal(70, 20, num_vbf))
    ])
    
    data['m_WZ'] = np.concatenate([
        np.abs(np.random.normal(320, 40, num_vbs)),
        np.abs(np.random.normal(290, 40, num_vbf))
    ])
    
    data['pt_ll'] = np.abs(np.random.normal(50, 20, num_events))
    data['delta_phi_ll'] = np.abs(np.random.uniform(0.5, 2.8, num_events))
    data['centrality_jj'] = np.random.uniform(0.4, 0.9, num_events)
    data['met'] = np.abs(np.random.exponential(45, num_events))
    data['n_jets'] = np.random.poisson(2.7, num_events)
    
    # Interference - more negative for VBS
    data['f_I'] = np.concatenate([
        np.random.normal(-0.6, 0.1, num_vbs),
        np.random.normal(-0.3, 0.2, num_vbf)
    ])
    
    # Mixed term
    data['f_TL'] = 0.6 * data['f_L'] + 0.4 * data['f_T']
    
    # Target - VBS=1, VBF=0
    data['is_vbs'] = np.concatenate([np.ones(num_vbs), np.zeros(num_vbf)])
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

# ================================================
# PHYSICS-BASED FEATURE ENGINEERING
# ================================================
def add_discriminative_features(df):
    """Add features that enhance VBS/VBF separation"""
    # Relative transverse momentum fraction
    df['pt_balance'] = df['pt_j1'] / (df['pt_j1'] + df['pt_j2'] + 1e-8)
    
    # Polarization dominance
    df['L_minus_T'] = df['f_L'] - df['f_T']
    
    # Jet-lepton centrality
    df['jet_lepton_centrality'] = df['centrality_jj'] * (1 - df['delta_phi_ll']/3.14)
    
    # Scattering angle proxy
    df['scattering_angle'] = np.log(df['m_jj'] / df['m_WZ'])
    
    return df

# ================================================
# MODEL ARCHITECTURE
# ================================================
def build_physics_model(input_dim):
    """Neural network with physics-inspired architecture"""
    inputs = Input(shape=(input_dim,))
    
    # Feature emphasis layer
    x = Dense(64, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Physics-informed layer
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Polarization attention layer
    x = Dense(64, activation='tanh')(x)
    
    # Output layer
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=output)
    return model

# ================================================
# PHYSICS-CONSTRAINED LOSS FUNCTION
# ================================================
def polarization_constrained_loss(y_true, y_pred):
    """Custom loss incorporating polarization physics"""
    # Standard binary crossentropy
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # Physics regularization:
    # 1. VBS should have high longitudinal polarization
    # 2. VBF should have high transverse polarization
    # Using y_pred as proxy for polarization strength
    physics_loss = 0.1 * K.mean(
        y_true * K.square(1 - y_pred) +   # Penalize low prob for VBS events
        (1 - y_true) * K.square(y_pred)    # Penalize high prob for VBF events
    )
    
    return bce + physics_loss

# ================================================
# DIAGNOSTIC VISUALIZATION
# ================================================
def plot_feature_discrimination(df):
    """Visualize key differences between VBS and VBF"""
    # Create a copy with converted target for plotting
    df_plot = df.copy()
    df_plot['is_vbs'] = df_plot['is_vbs'].astype(int)
    
    plt.figure(figsize=(15, 12))
    
    # Plot 1: m_jj vs delta_eta_jj
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=df_plot, x='m_jj', y='delta_eta_jj', hue='is_vbs', 
                    palette={0: 'blue', 1: 'red'}, alpha=0.6)
    plt.title('Dijet Mass vs Pseudorapidity Separation')
    plt.axvline(600, color='k', linestyle='--', alpha=0.5)
    plt.axhline(4.0, color='k', linestyle='--', alpha=0.5)
    plt.annotate('VBS Region', (700, 5.0), fontsize=12)
    
    # Plot 2: Longitudinal vs Transverse polarization
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df_plot, x='f_L', y='f_T', hue='is_vbs', 
                    palette={0: 'blue', 1: 'red'}, alpha=0.6)
    plt.title('Longitudinal vs Transverse Polarization')
    plt.plot([0, 1], [1, 0], 'k--', alpha=0.5)
    plt.annotate('VBS: High L, Low T', (0.8, 0.1), fontsize=12)
    plt.annotate('VBF: Low L, High T', (0.1, 0.8), fontsize=12)
    
    # Plot 3: L-T polarization difference
    plt.subplot(2, 2, 3)
    sns.histplot(data=df_plot, x='L_minus_T', hue='is_vbs', 
                 element='step', stat='density', common_norm=False, 
                 palette={0: 'blue', 1: 'red'})
    plt.title('Longitudinal - Transverse Polarization Difference')
    plt.axvline(0, color='k', linestyle='--', alpha=0.5)
    plt.annotate('VBS: L > T', (0.3, 1.5), fontsize=12)
    plt.annotate('VBF: T > L', (-0.3, 1.5), fontsize=12)
    
    # Plot 4: Scattering angle - FIXED with proper hue handling
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df_plot, x='is_vbs', y='scattering_angle', 
                hue='is_vbs', palette={0: 'blue', 1: 'red'}, legend=False)
    plt.title('Scattering Angle Proxy')
    plt.xticks([0, 1], ['VBF', 'VBS'])
    plt.xlabel('Process')
    
    plt.tight_layout()
    plt.savefig('feature_discrimination.png', dpi=150)
    plt.show()

# ================================================
# MAIN EXECUTION
# ================================================
if __name__ == "__main__":
    # Step 1: Generate physics-based data
    print("Generating physics-based dataset...")
    df = generate_physics_data(50000)
    
    # Step 2: Add discriminative features
    print("Adding physics-based features...")
    df = add_discriminative_features(df)
    
    # Step 3: Visualize feature discrimination
    print("Plotting feature discrimination...")
    plot_feature_discrimination(df)
    
    # Step 4: Prepare features
    print("Preparing features and target...")
    features = [
        'm_jj', 'delta_eta_jj', 'pt_j1', 'pt_j2', 'm_WZ', 
        'pt_ll', 'delta_phi_ll', 'centrality_jj', 'met', 'n_jets',
        'f_L', 'f_T', 'f_I', 'f_TL',
        'pt_balance', 'L_minus_T', 'jet_lepton_centrality', 'scattering_angle'
    ]
    
    X = df[features]
    y = df['is_vbs']
    
    # Step 5: Preprocessing
    print("Preprocessing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Step 6: Build model
    print("Building physics-constrained model...")
    model = build_physics_model(X_train.shape[1])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=polarization_constrained_loss,
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    model.summary()
    
    # Step 7: Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=256,
        callbacks=[
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
        ],
        verbose=1
    )
    
    # Step 8: Evaluate model
    print("\nEvaluating model...")
    y_pred = model.predict(X_test).ravel()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(tpr, 1-fpr, 'b-', lw=2, label=f'Model (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [1, 0], 'k--')
    plt.xlabel('VBS Efficiency (True Positive Rate)')
    plt.ylabel('1 - VBF Efficiency (Background Rejection)')
    plt.title('VBS vs VBF Classification ROC')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.savefig('roc_curve.png', dpi=150)
    plt.show()
    
    # Calculate performance metrics
    vbs_eff = np.mean(y_pred[y_test==1] > 0.5)
    vbf_rejection = np.mean(y_pred[y_test==0] <= 0.5)
    
    print(f"VBS Detection Efficiency: {vbs_eff:.4f}")
    print(f"VBF Rejection Efficiency: {vbf_rejection:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Step 9: Save model and artifacts
    print("\nSaving model and artifacts...")
    model.save('vbs_vbf_classifier.keras')
    np.save('vbs_scaler.npy', {
        'mean': scaler.mean_,
        'scale': scaler.scale_,
        'features': features
    })
    
    # Step 10: Sample predictions
    print("\nSample predictions:")
    sample_vbs = X_test[y_test == 1][0]
    sample_vbf = X_test[y_test == 0][0]
    
    for name, sample in [('VBS', sample_vbs), ('VBF', sample_vbf)]:
        sample = sample.reshape(1, -1)
        prob = model.predict(sample, verbose=0)[0][0]
        print(f"{name} sample prediction: {prob:.4f} ({'VBS' if prob > 0.5 else 'VBF'})")
    
    print("\nTraining complete! Key artifacts saved:")
    print("- vbs_vbf_classifier.keras (model)")
    print("- vbs_scaler.npy (scaler parameters)")
    print("- feature_discrimination.png")
    print("- roc_curve.png")
