import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler  # Added missing import

# ================================================
# LOAD TRAINED MODEL AND SCALER
# ================================================
def load_artifacts():
    """Load trained model and scaler"""
    try:
        # Load model with custom loss function
        model = load_model('vbs_vbf_classifier.keras', 
                          custom_objects={'polarization_constrained_loss': lambda y_true, y_pred: y_pred},
                          compile=False)  # Disable compilation for prediction
        
        # Load scaler parameters
        scaler_params = np.load('vbs_scaler.npy', allow_pickle=True).item()
        features = scaler_params['features']
        
        return model, features, scaler_params
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return None, None, None

# ================================================
# USER INPUT INTERFACE
# ================================================
def get_user_input(features):
    """Prompt user for feature values"""
    input_data = {}
    print("\n" + "="*50)
    print("VBS/VBF CLASSIFICATION INTERFACE")
    print("="*50)
    print("Please enter the following feature values:")
    
    # Define units for better user experience
    units = {
        'm_jj': 'GeV', 'delta_eta_jj': '', 'pt_j1': 'GeV', 'pt_j2': 'GeV',
        'm_WZ': 'GeV', 'pt_ll': 'GeV', 'delta_phi_ll': 'radians',
        'centrality_jj': '', 'met': 'GeV', 'n_jets': '',
        'f_L': '(0-1)', 'f_T': '(0-1)', 'f_I': '', 'f_TL': '',
        'pt_balance': '', 'L_minus_T': '', 'jet_lepton_centrality': '',
        'scattering_angle': ''
    }
    
    # Get input for each feature
    for feature in features:
        unit = units.get(feature, '')
        while True:
            try:
                value = float(input(f"  • {feature} {unit}: "))
                input_data[feature] = value
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
    
    return input_data

# ================================================
# PHYSICS-BASED FEATURE ENGINEERING (FOR REAL-TIME)
# ================================================
def add_features_realtime(input_dict):
    """Add engineered features to user input"""
    # Create a copy to avoid modifying original
    data = input_dict.copy()
    
    # Add engineered features
    data['pt_balance'] = data['pt_j1'] / (data['pt_j1'] + data['pt_j2'] + 1e-8)
    data['L_minus_T'] = data['f_L'] - data['f_T']
    data['jet_lepton_centrality'] = data['centrality_jj'] * (1 - data['delta_phi_ll']/3.14)
    data['scattering_angle'] = np.log(data['m_jj'] / data['m_WZ'])
    
    return data

# ================================================
# MAIN PREDICTION FUNCTION
# ================================================
def predict_vbs_vbf():
    """Main function to handle user input and prediction"""
    # Load model and scaler
    model, features, scaler_params = load_artifacts()
    if model is None:
        print("Failed to load model. Please train the model first.")
        return
    
    # Get user input
    input_data = get_user_input(features)
    
    # Add engineered features
    full_data = add_features_realtime(input_data)
    
    # Create DataFrame with correct feature order
    input_df = pd.DataFrame([full_data])[features]
    
    # Scale features
    scaler = StandardScaler()
    scaler.mean_ = scaler_params['mean']
    scaler.scale_ = scaler_params['scale']
    scaled_input = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(scaled_input)[0][0]
    vbs_prob = prediction * 100
    vbf_prob = 100 - vbs_prob
    
    # Print results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"VBS Probability: {vbs_prob:.2f}%")
    print(f"VBF Probability: {vbf_prob:.2f}%")
    print("-"*50)
    
    if prediction > 0.5:
        print("CLASSIFICATION: VECTOR BOSON SCATTERING (VBS)")
        print("Characteristics expected:")
        print("- High dijet mass (>600 GeV)")
        print("- Large pseudorapidity separation (>4.0)")
        print("- Longitudinal polarization dominance (f_L > f_T)")
    else:
        print("CLASSIFICATION: VECTOR BOSON FUSION (VBF)")
        print("Characteristics expected:")
        print("- Moderate dijet mass (∼500 GeV)")
        print("- Smaller pseudorapidity separation (<4.0)")
        print("- Transverse polarization dominance (f_T > f_L)")
    
    print("="*50)

# ================================================
# RUN THE APPLICATION
# ================================================
if __name__ == "__main__":
    # Run the prediction interface
    predict_vbs_vbf()
    
    # Option to run again
    while True:
        again = input("\nWould you like to classify another event? (y/n): ").lower()
        if again == 'y':
            predict_vbs_vbf()
        else:
            print("Exiting VBS/VBF classification system.")
            break