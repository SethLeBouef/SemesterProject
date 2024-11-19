# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from src.config import Config

# Initialize the connection
config = Config()
engine = config.get_engine()

# Query the database to retrieve required columns from the updated table
query = """
SELECT subject, sessionIndex, rep, avg_dwell_time AS dwell_time 
FROM KeyLoggerFull.Features_Created;
"""
data = pd.read_sql(query, con=engine)

# Calculate the average dwell time per user
average_total_dwell_time_per_user = (
    data.groupby('subject')['dwell_time']
    .mean()
    .sort_values()
)

# Plot the graph with precision improvements
plt.figure(figsize=(12, 6))
bars = average_total_dwell_time_per_user.plot(kind='bar', color='teal', width=0.8)

# Adjust the Y axis values based on your data's dwell time range
plt.yticks(
    ticks=[0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
    fontsize=10
)

# Add data labels to the graph
for bar in bars.patches:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2, height + 0.02,
        f'{height:.4f}', ha='center', va='bottom', fontsize=9, rotation=90,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

# Customize the graph
plt.xlabel('User ID')
plt.ylabel('Average Dwell Time per Password Entry (Seconds)')
plt.title('Precise Average Dwell Time for Full Password Entry per User')
plt.xticks(rotation=90)  # Rotate x-axis labels for clarity
plt.grid(axis='y', linestyle='--', alpha=1)

# Display the graph
plt.show()