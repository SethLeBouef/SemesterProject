import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.config import Config

config = Config()
engine = config.get_engine()
# Define features
features = [
    'avg_flight_time', 'avg_dwell_time', 'total_typing_duration', 'hold_ratio', 'CPS', 'WPM',
    'std_flight_time', 'std_dwell_time', 'pause_ratio', 'total_typing_duration_mean',
    'total_typing_duration_std', 'avg_flight_time_mean', 'avg_flight_time_std',
    'avg_dwell_time_mean', 'avg_dwell_time_std', 'CPS_mean', 'CPS_std', 'WPM_mean', 'WPM_std',
    'flight_to_dwell_ratio', 'typing_duration_per_character', 'typing_efficiency'
]

# Load the Feature_Creation database
#  Load Data

#  Load Data
query = """
SELECT * FROM Features_Created
"""
data = pd.read_sql_query(query, con=engine)


# Ensure target variable is correctly identified (replace 'target' with the actual target column name)
target = 'WPM_mean'
X = data[features]
y = data[target]

# 1. Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Features")
plt.show()

# 2. Feature Distribution Plots
for feature in features:
    plt.figure(figsize=(6, 4))
    sns.histplot(data[feature], kde=True, bins=30)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()

# 3. Boxplots for Outlier Detection
for feature in features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=data[feature])
    plt.title(f"Boxplot of {feature}")
    plt.xlabel(feature)
    plt.show()

# 4. Pair Plot (Optional for Selected Features)
selected_features = ['avg_flight_time', 'avg_dwell_time', 'WPM', 'CPS']
sns.pairplot(data[selected_features])
plt.title("Pair Plot of Selected Features")
plt.show()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 5. ROC Curve
y_pred_proba = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "r--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# 6. Confusion Matrix
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()