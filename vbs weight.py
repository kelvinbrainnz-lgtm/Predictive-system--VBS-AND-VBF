import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import warnings

# ================================================
# ENVIRONMENT SETUP: Suppress warnings and logs
# ================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info/warning logs
tf.get_logger().setLevel('ERROR')  # Suppress TF logging output
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# ================================================
# MODEL WEIGHT ANALYSIS FUNCTION
# ================================================
def analyze_model_weights(model_path, scaler_path):
    """Analyze and visualize model weights for interpretability"""
    print("\n" + "="*50)
    print("MODEL WEIGHT ANALYSIS")
    print("="*50)
    
    # Load model
    try:
        def dummy_loss(y_true, y_pred):
            return y_pred

        model = load_model(model_path, custom_objects={
            'polarization_constrained_loss': dummy_loss
        }, compile=False)
        print("Model loaded successfully")
    except Exception as e:
        print(f" Error loading model: {e}")
        return None, None
    
    # Load scaler to get feature names
    try:
        scaler_params = np.load(scaler_path, allow_pickle=True).item()
        features = scaler_params['features']
        print(f" Found {len(features)} features in scaler")
    except Exception as e:
        print(f" Error loading scaler: {e}")
        features = None
    
    # Print model architecture
    print("\nModel Architecture:")
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: {layer.name} ({layer.__class__.__name__})")
        weights = layer.get_weights()
        if not weights:
            print("  No weights")
            continue
        for j, w in enumerate(weights):
            print(f"  Weight {j} shape: {w.shape}")
    
    # Identify the first dense layer connected to input
    first_dense = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            weights = layer.get_weights()
            if weights and weights[0].shape[0] == len(features):
                first_dense = layer
                break

    if first_dense:
        print("\n Analyzing first dense layer weights...")
        weights, biases = first_dense.get_weights()

        avg_abs_weights = np.mean(np.abs(weights), axis=1)

        if features:
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': avg_abs_weights
            }).sort_values('Importance', ascending=False)

            print("\n Feature Importance (First Layer):")
            print(feature_importance.to_string(index=False))

            # Bar plot
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
            plt.title('Feature Importance in First Dense Layer')
            plt.xlabel('Average Absolute Weight Magnitude')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=150)
            plt.show()
        else:
            print("No feature names available. Showing raw weight magnitudes:")
            print(avg_abs_weights)

        # Plot distribution of weights for each feature
        if features:
            plt.figure(figsize=(14, 10))
            n_features = len(features)
            n_cols = 4
            n_rows = (n_features + n_cols - 1) // n_cols

            for i, feature in enumerate(features):
                plt.subplot(n_rows, n_cols, i+1)
                sns.histplot(weights[i], bins=30, kde=True, color='skyblue')
                plt.title(f"{feature[:15]}{'...' if len(feature) > 15 else ''}")
                plt.xlabel('Weight Value')
                plt.ylabel('Frequency')

            plt.tight_layout()
            plt.savefig('weight_distributions.png', dpi=150)
            plt.show()

        return weights, biases
    else:
        print(" Could not find first dense layer for feature analysis")
        return None, None

# ================================================
# MAIN EXECUTION
# ================================================
if __name__ == "__main__":
    MODEL_PATH = 'vbs_vbf_classifier.keras'
    SCALER_PATH = 'vbs_scaler.npy'

    weights, biases = analyze_model_weights(MODEL_PATH, SCALER_PATH)

    if weights is not None:
        print("\n Additional Weight Statistics:")
        print(f"Total weights in first layer: {weights.size}")
        print(f"Average weight magnitude: {np.mean(np.abs(weights)):.4f}")
        print(f"Max positive weight: {np.max(weights):.4f}")
        print(f"Max negative weight: {np.min(weights):.4f}")

        plt.figure(figsize=(10, 6))
        sns.histplot(weights.flatten(), bins=50, kde=True, color='purple')
        plt.title('Overall Weight Distribution in First Layer')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('overall_weight_distribution.png', dpi=150)
        plt.show()

        
    