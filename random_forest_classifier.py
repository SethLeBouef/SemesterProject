# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.utils import shuffle
import shap

# Load your dataset 
try:
    from src.config import Config
    config = Config()
    engine = config.get_engine()
except ImportError:
    raise ImportError("Ensure 'Config' is properly defined and 'get_engine()' provides a valid SQLAlchemy engine.")

# Query with the best features
query = """
SELECT subject, 
       avg_flight_time, avg_dwell_time, total_typing_duration, hold_ratio, CPS, WPM,
       std_flight_time, std_dwell_time, pause_ratio, total_typing_duration_mean,
       total_typing_duration_std, avg_flight_time_mean, avg_flight_time_std,
       avg_dwell_time_mean, avg_dwell_time_std, CPS_mean, CPS_std, WPM_mean, WPM_std,
       flight_to_dwell_ratio, typing_duration_per_character, typing_efficiency
FROM Features_Created;
"""
data = pd.read_sql_query(query, con=engine)

# Check for missing values
if data.isnull().any().any():
    print("Missing values detected. Filling with median values.")
    data.fillna(data.median(), inplace=True)

# Define features and target
features = [
    'avg_flight_time', 'avg_dwell_time', 'total_typing_duration', 'hold_ratio', 'CPS', 'WPM',
    'std_flight_time', 'std_dwell_time', 'pause_ratio', 'total_typing_duration_mean',
    'total_typing_duration_std', 'avg_flight_time_mean', 'avg_flight_time_std',
    'avg_dwell_time_mean', 'avg_dwell_time_std', 'CPS_mean', 'CPS_std', 'WPM_mean', 'WPM_std',
    'flight_to_dwell_ratio', 'typing_duration_per_character', 'typing_efficiency'
]
X = data[features]
y = data['subject']

# Shuffle the data 
X, y = shuffle(X, y, random_state=123)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)

# Define the Random Forest pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=200, 
        max_depth=20, 
        min_samples_split=5, 
        min_samples_leaf=3, 
        random_state=123,
        class_weight="balanced"
    ))
])

# Fit the model
pipeline.fit(X_train, y_train)

# Evaluate on the test set
y_pred = pipeline.predict(X_test)
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred, zero_division=1))
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))

# Cross-validation for overfitting check
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")

print("\nCross-Validation Accuracy: {:.2f}% ± {:.2f}%".format(cv_scores.mean() * 100, cv_scores.std() * 100))

# Check permutation importance
perm_importance = permutation_importance(
    pipeline.named_steps["classifier"], 
    X_test, 
    y_test, 
    n_repeats=10, 
    random_state=123
)

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": perm_importance.importances_mean
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance (Permutation):")
print(importance_df)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Permutation Feature Importance")
plt.show()

# Validate no overfitting with learning curve
train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8]
train_scores = []
test_scores = []

for size in train_sizes:
    X_partial, _, y_partial, _ = train_test_split(X_train, y_train, train_size=size, random_state=123, stratify=y_train)
    pipeline.fit(X_partial, y_partial)
    train_scores.append(accuracy_score(y_partial, pipeline.predict(X_partial)))
    test_scores.append(accuracy_score(y_test, pipeline.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores, marker="o", label="Train Accuracy")
plt.plot(train_sizes, test_scores, marker="o", label="Test Accuracy")
plt.title("Learning Curve")
plt.xlabel("Training Size Proportion")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# SHAP for feature explanation
explainer = shap.TreeExplainer(pipeline.named_steps["classifier"])
X_test_transformed = pipeline.named_steps['scaler'].transform(X_test)  # Get the transformed data
shap_values = explainer.shap_values(X_test_transformed)

# Ensure SHAP values align with X_test_transformed
shap.summary_plot(shap_values[0], X_test_transformed, feature_names=features, plot_type="bar")