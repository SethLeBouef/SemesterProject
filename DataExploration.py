import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

from src.config import Config
config = Config()
engine = config.get_engine()

query = "SELECT * FROM Features_Created"

# Execute the query and load data into a Pandas DataFrame
features_data = pd.read_sql(query, engine)


# Select features for analysis
features_to_plot = ['avg_dwell_time', 'avg_flight_time', 'total_typing_duration', 'CPS', 'WPM']

# === Visualization: Histograms (Feature Distributions) ===
for feature in features_to_plot:
    plt.figure(figsize=(8, 5))
    sns.histplot(features_data[feature], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# Interpretation: Neural networks can adapt to skewed and non-normal distributions.
# Random Forests may struggle with overfitting if distributions are skewed.

# === Visualization: Box Plots (Outliers) ===
plt.figure(figsize=(10, 6))
sns.boxplot(data=features_data[features_to_plot])
plt.title('Box Plots of Key Features')
plt.xticks(rotation=45)
plt.show()

# Interpretation: Random Forests are sensitive to outliers, leading to overfitting. 
# Neural networks, with proper regularization, are less affected by outliers.

# === Visualization: Scatter Plots (Relationships Between Features) ===
scatter_pairs = [('avg_dwell_time', 'avg_flight_time'), 
                 ('CPS', 'WPM'), 
                 ('total_typing_duration', 'CPS')]

for x, y in scatter_pairs:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=features_data[x], y=features_data[y])
    plt.title(f'Scatter Plot: {x} vs {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

# Interpretation: Random Forests work well with linear relationships but struggle with complex, nonlinear patterns. 
# Neural networks excel in capturing nonlinear relationships.

# === Visualization: Correlation Matrix (Multicollinearity) ===
correlation_matrix = features_data[features_to_plot].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.show()

# Interpretation: Random Forests may perform redundant splits when features are highly correlated.
# Neural networks can handle multicollinearity better by learning combined feature representations.

# === PCA: Dimensionality Analysis ===
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_data[features_to_plot])

pca = PCA(n_components=0.95)  # Retain 95% variance
pca_result = pca.fit_transform(scaled_features)

print(f"Original number of features: {len(features_to_plot)}")
print(f"Number of components explaining 95% variance: {pca.n_components_}")

# Visualization: Cumulative Variance Explained by Principal Components
pca_full = PCA().fit(scaled_features)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('Cumulative Variance Explained by Principal Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.grid()
plt.show()

# Interpretation: Neural networks handle high-dimensional data well, while Random Forests may struggle 
# with too many irrelevant or redundant features.

# === Nonlinearity Detection ===
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Test nonlinearity for a pair of features
x = features_data['avg_dwell_time'].values.reshape(-1, 1)
y = features_data['avg_flight_time'].values

linear_model = LinearRegression()
linear_model.fit(x, y)
predictions = linear_model.predict(x)

# Scatter Plot with Linear Fit
plt.figure(figsize=(8, 5))
sns.scatterplot(x=features_data['avg_dwell_time'], y=features_data['avg_flight_time'], label='Data')
plt.plot(features_data['avg_dwell_time'], predictions, color='red', label='Linear Fit')
plt.title('Nonlinearity Check: avg_dwell_time vs avg_flight_time')
plt.xlabel('avg_dwell_time')
plt.ylabel('avg_flight_time')
plt.legend()
plt.show()

mse = mean_squared_error(y, predictions)
print(f"Mean Squared Error of Linear Fit: {mse}")

# Interpretation: If linear fit has a high error, it suggests nonlinearity, which neural networks can model better than Random Forests.