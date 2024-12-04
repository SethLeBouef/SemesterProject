# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib  # For saving the scaler
import pickle

#  Initialize the database connection
from src.config import Config
config = Config()
engine = config.get_engine()

#  Load Data
query = """
SELECT * FROM Features_Created
"""
data = pd.read_sql_query(query, con=engine)

#  Select Features and Target
features = [
    'avg_flight_time', 'avg_dwell_time', 'total_typing_duration', 'hold_ratio', 'CPS', 'WPM',
    'std_flight_time', 'std_dwell_time', 'pause_ratio', 'total_typing_duration_mean',
    'total_typing_duration_std', 'avg_flight_time_mean', 'avg_flight_time_std',
    'avg_dwell_time_mean', 'avg_dwell_time_std', 'CPS_mean', 'CPS_std', 'WPM_mean', 'WPM_std',
    'flight_to_dwell_ratio', 'typing_duration_per_character', 'typing_efficiency'
]
X = data[features]
y = data['subject']

# Encode target variable (subjects) into integers
y_encoded = pd.factorize(y)[0]

#  Split Data into Train, Test, and Holdout Sets
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

#  Standardize the Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_holdout = scaler.transform(X_holdout)

# Save the scaler for later use
joblib.dump(scaler, 'scaler.pkl')

#  Build the Neural Network
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y_encoded)), activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks: Early Stopping and Learning Rate Scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Train the Model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, lr_schedule],
    verbose=1
)

# Save training history
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

#  Evaluate the Model on Holdout Test Set
holdout_loss, holdout_accuracy = model.evaluate(X_holdout, y_holdout)
print(f"\nHoldout Test Accuracy: {holdout_accuracy * 100:.2f}%")

#  Check Overfitting with Learning Curves
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Learning Curve: Train vs Validation Accuracy')
plt.legend()
plt.show()

#  Confusion Matrix and Classification Report
y_pred = model.predict(X_holdout)
y_pred_labels = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_holdout, y_pred_labels)
ConfusionMatrixDisplay(cm).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:\n", classification_report(y_holdout, y_pred_labels, zero_division=1))

#  Test  Noisy Data
noisy_X_holdout = X_holdout + np.random.normal(0, 0.01, X_holdout.shape)
noisy_loss, noisy_accuracy = model.evaluate(noisy_X_holdout, y_holdout)
print(f"\nAccuracy on Noisy Data: {noisy_accuracy * 100:.2f}%")

#  Evaluate using Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_index, test_index) in enumerate(kf.split(X, y_encoded)):
    print(f"\nFold {fold + 1}")
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y_encoded[train_index], y_encoded[test_index]

    # Standardize the data
    fold_scaler = StandardScaler()
    X_train_scaled = fold_scaler.fit_transform(X_train_fold)
    X_test_scaled = fold_scaler.transform(X_test_fold)

    # Train the model on this fold
    history_fold = model.fit(
        X_train_scaled, y_train_fold,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, lr_schedule],
        verbose=0
    )

    # Evaluate on the fold
    fold_loss, fold_accuracy = model.evaluate(X_test_scaled, y_test_fold, verbose=0)
    fold_accuracies.append(fold_accuracy)
    print(f"Fold Accuracy: {fold_accuracy * 100:.2f}%")

print(f"\nAverage Cross-Validation Accuracy: {np.mean(fold_accuracies) * 100:.2f}%")

# Save the Model
model.save('keystroke_model.keras')  # Save in the Keras format