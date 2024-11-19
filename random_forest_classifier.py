# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.config import Config

# Initialize the database connection
config = Config()
engine = config.get_engine()

# Load data from the Features_Created table
query = """
SELECT 
    subject, 
    total_typing_duration, 
    hold_ratio, 
    CPS, 
    WPM, 
    avg_flight_time, 
    avg_dwell_time, 
    pause_ratio, 
    std_flight_time, 
    std_dwell_time 
FROM Features_Created;
"""
session_data = pd.read_sql_query(query, con=engine)

# Check for missing or invalid values and handle them
if session_data.isnull().any().any():
    print("Missing values detected. Filling with median values.")
    session_data.fillna(session_data.median(), inplace=True)

# Select features and target
features = [
    'total_typing_duration', 
    'hold_ratio', 
    'CPS', 
    'WPM', 
    'avg_flight_time', 
    'avg_dwell_time', 
    'pause_ratio', 
    'std_flight_time', 
    'std_dwell_time'
]
X = session_data[features]
y = session_data['subject']

# Define the pipeline with scaling and the RandomForest classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=123, class_weight='balanced'))
])

# Parameter grid for GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200, 500],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

# Define scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted', zero_division=1),
    'recall': make_scorer(recall_score, average='weighted', zero_division=1),
    'f1': make_scorer(f1_score, average='weighted', zero_division=1)
}

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X, y)

# Retrieve the best model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Split data for test evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

# Fit best model on training data
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Model Test Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
print("\nClassification Report on Test Set:\n", classification_report(y_test, y_pred, zero_division=1))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=best_model.classes_,
            yticklabels=best_model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix of Tuned Random Forest Model on Test Set")
plt.show()