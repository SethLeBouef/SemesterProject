import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from collections import Counter

# Load the trained model
model = load_model('keystroke_model.h5')

# Load the s scaler
scaler = joblib.load('scaler.pkl')

# Feature variables
features = [
    'avg_flight_time', 'avg_dwell_time', 'total_typing_duration', 'hold_ratio', 'CPS', 'WPM',
    'std_flight_time', 'std_dwell_time', 'pause_ratio', 'total_typing_duration_mean',
    'total_typing_duration_std', 'avg_flight_time_mean', 'avg_flight_time_std',
    'avg_dwell_time_mean', 'avg_dwell_time_std', 'CPS_mean', 'CPS_std', 'WPM_mean', 'WPM_std',
    'flight_to_dwell_ratio', 'typing_duration_per_character', 'typing_efficiency'
]

# Load n input data
def load_data(engine, table_name='Features_Created'):
    """Load data from the database."""
    query = f"SELECT * FROM {table_name}"
    data = pd.read_sql_query(query, con=engine)
    return data

# Preprocess data
def preprocess_data(data):
    
    try:
        X = data[features]
        X_scaled = scaler.transform(X)  # Use the saved scaler to standardize
        print("Preprocessing successful. Sample scaled data:\n", X_scaled[:5])
        return X_scaled
    except KeyError as e:
        print(f"Missing feature in the data: {e}")
        raise
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise

# Predict with the model
def make_predictions(model, X_scaled, confidence_threshold=0.8):
    
    predictions = model.predict(X_scaled)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Filter predictions by confidence
    confident_predictions = [
        (cls, max(prob)) for cls, prob in zip(predicted_classes, predictions)
        if max(prob) >= confidence_threshold
    ]
    predicted_classes = [cls for cls, _ in confident_predictions]
    confidences = [conf for _, conf in confident_predictions]
    
    print(f"Confident predictions: {len(predicted_classes)} / {len(X_scaled)} samples")
    return predicted_classes, confidences, predictions

# Main function
def main(engine):
    """Load data, preprocess, and make predictions."""
    # Load data from the database
    data = load_data(engine)
    
    # Preprocess the data
    X_scaled = preprocess_data(data)
    
    # Make predictions
    confidence_threshold = 0.8  # Adjust as necessary
    predicted_classes, confidences, predictions = make_predictions(model, X_scaled, confidence_threshold)
    
    # Map predicted classes back to labels
    unique_classes = np.unique(predicted_classes)
    subject_mapping = {cls: f"Subject_{cls}" for cls in unique_classes}  # Ensure valid mapping
    
    # Avoid KeyError with a default value for unmapped classes
    predicted_labels = [subject_mapping.get(cls, f"Unknown_{cls}") for cls in predicted_classes]
    
    # Debugging for mapping
    print("Subject mapping:", subject_mapping)
    
    # Print predictions
    for i, (label, confidence) in enumerate(zip(predicted_labels, confidences)):
        print(f"Sample {i + 1}: Predicted Class: {label} (Confidence: {confidence:.2f})")
    
    # Summary statistics
    print("Predicted class distribution:", Counter(predicted_classes))
    
    # Return predictions
    return predicted_classes, confidences, predictions

# Run the program
if __name__ == "__main__":
    from src.config import Config  # Import your Config for database connection
    config = Config()
    engine = config.get_engine()
    
    main(engine)