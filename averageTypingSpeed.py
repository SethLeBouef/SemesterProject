# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import Config

# Initialize the database connection
config = Config()
engine = config.get_engine()

# Query the data from Features_Created
query = """
SELECT 
    subject, 
    sessionIndex, 
    rep, 
    total_characters, 
    avg_dwell_time, 
    avg_flight_time, 
    CPS, 
    WPM 
FROM Features_Created;
"""
data = pd.read_sql(query, con=engine)


# Calculate the average typing speed per user (CPS and WPM)
average_speed_per_user = data.groupby('subject').agg(
    average_CPS=('CPS', 'mean'),
    average_WPM=('WPM', 'mean')
)

# Display the average typing speeds
print("Average Typing Speeds (CPS and WPM) per User:\n")
print(average_speed_per_user)

### Bar Chart

# Plot the average WPM for each user
plt.figure(figsize=(12, 6))
bars = average_speed_per_user['average_WPM'].plot(kind='bar', color='purple', width=0.8)

# Add data labels to the bars
for bar in bars.patches:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2, height + 0.1,
        f'{height:.2f}', ha='center', va='bottom', fontsize=9, rotation=90,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

# Customize the plot
plt.xlabel('User ID')
plt.ylabel('Average WPM (Words Per Minute)')
plt.title('Average Typing Speed (WPM) per User')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()
###BOX PLOT
"""
plt.figure(figsize=(10, 6))
data.boxplot(column='WPM', by='subject')
plt.title('WPM Distribution per User')
plt.suptitle('')  # Removes the default title
plt.xlabel('User ID')
plt.xticks(rotation=90)
plt.ylabel('Words Per Minute (WPM)')
plt.show()
"""


###Scatter Plot
"""
plt.figure(figsize=(10, 6))
sns.regplot(x='total_dwell_time', y='WPM', data=data, scatter_kws={'alpha':.2})
plt.title('Relationship between Dwell Time and WPM')
plt.xlabel('Total Dwell Time (Seconds)')
plt.ylabel('Words Per Minute (WPM)')
plt.show()

"""



###SwarmPlot
"""
plt.figure(figsize=(12, 6))
sns.swarmplot(x='subject', y='WPM', data=data, color='purple', size=.75)  # Smaller size for large datasets

# Customize plot appearance
plt.title('Individual WPM Values per User')
plt.xlabel('User ID')
plt.ylabel('Words Per Minute (WPM)')
plt.xticks(rotation=90)  # Rotate x-axis labels for clarity
plt.show()
"""

###HeatMap that is useful
"""
# Aggregate WPM by sessionIndex and subject if there are multiple entries per combination
data = data.groupby(['sessionIndex', 'subject'], as_index=False)['WPM'].mean()
# Pivot the data so that sessionIndex is the row index, subject is the column, and WPM is the value
heatmap_data = data.pivot(index='sessionIndex', columns='subject', values='WPM')

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, annot_kws={'size': 8, 'rotation':90}, fmt=".1f", cbar_kws={'label': 'WPM'})
plt.title('WPM per User Across Sessions')
plt.xlabel('User ID')
plt.ylabel('Session Index')
plt.show()

"""
