import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib  # For loading the scaler

# Load the trained model
model = load_model('keystroke_model.h5')

# Load the saved scaler
scaler = joblib.load('scaler.pkl')

# Define the feature columns (must match the training features)
features = [
    'avg_flight_time', 'avg_dwell_time', 'total_typing_duration', 'hold_ratio', 'CPS', 'WPM',
    'std_flight_time', 'std_dwell_time', 'pause_ratio', 'total_typing_duration_mean',
    'total_typing_duration_std', 'avg_flight_time_mean', 'avg_flight_time_std',
    'avg_dwell_time_mean', 'avg_dwell_time_std', 'CPS_mean', 'CPS_std', 'WPM_mean', 'WPM_std',
    'flight_to_dwell_ratio', 'typing_duration_per_character', 'typing_efficiency'
]

# Load new input data
def load_data(engine, table_name='Features_Created'):
    """Load data from the database."""
    query = f"SELECT * FROM {table_name}"
    data = pd.read_sql_query(query, con=engine)
    return data

# Preprocess data
def preprocess_data(data):
    """Preprocess the input data to match the model's requirements."""
    X = data[features]
    X_scaled = scaler.transform(X)  # Use the saved scaler to standardize
    return X_scaled

# Predict with the model
def make_predictions(model, X_scaled):
    """Make predictions using the trained model."""
    predictions = model.predict(X_scaled)
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes, predictions

# Main function
def main(engine):
    """Load data, preprocess, and make predictions."""
    # Load new data
    data = load_data(engine)
    
    # Preprocess the data
    X_scaled = preprocess_data(data)
    
    # Make predictions
    predicted_classes, predictions = make_predictions(model, X_scaled)
    
    # Map predicted classes back to labels if available
    subject_mapping = {i: f"Subject_{i}" for i in range(len(np.unique(predicted_classes)))}  # Adjust with your labels
    predicted_labels = [subject_mapping[cls] for cls in predicted_classes]
    
    # Print predictions
    for i, label in enumerate(predicted_labels):
        print(f"Sample {i + 1}: Predicted Class: {label}")
    
    # Return predictions
    return predicted_classes, predictions

# Run the program
if __name__ == "__main__":
    from src.config import Config  # Import your Config for database connection
    config = Config()
    engine = config.get_engine()
    
    main(engine)