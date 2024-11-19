Used 

# Import necessary libraries

import pandas as pd
from sqlalchemy import create_engine
from src.config import Config

# Initialize the database connection
config = Config()
engine = config.get_engine()

# Step 1: Load Data from Keystroke_Raw Table
query = """
SELECT subject, sessionIndex, rep, `DD.period.t`, `DD.t.i`, `DD.i.e`, `DD.e.five`, 
       `DD.five.Shift.r`, `DD.Shift.r.o`, `DD.o.a`, `DD.a.n`, `DD.n.l`, `DD.l.Return` 
FROM KeyLoggerFull.Features_Created
"""
keystroke_data = pd.read_sql(query, con=engine)

# Step 2: Calculate Flight Times
# Create a list of all "DD" columns representing time between keystrokes
flight_time_columns = ['DD.period.t', 'DD.t.i', 'DD.i.e', 'DD.e.five', 
                       'DD.five.Shift.r', 'DD.Shift.r.o', 'DD.o.a', 
                       'DD.a.n', 'DD.n.l', 'DD.l.Return']

# Calculate individual flight times (if not averaged) and the average flight time for each subject-session
keystroke_data['calculated_avg_flight_time'] = keystroke_data[flight_time_columns].mean(axis=1)

# Step 3: Aggregate average flight time per user for comparison
average_flight_time_per_user = keystroke_data.groupby('subject')['calculated_avg_flight_time'].mean()

# Load the existing Features_Created table for comparison
query_features = """
SELECT subject, avg_flight_time 
FROM KeyLoggerFull.Features_Created
"""
features_data = pd.read_sql_query(query_features, con=engine)
features_avg_flight_time_per_user = features_data.groupby('subject')['avg_flight_time'].mean()

# Step 4: Compare Recalculated Flight Times with Stored Flight Times
comparison_df = pd.DataFrame({
    'Recalculated_Avg_Flight_Time': average_flight_time_per_user,
    'Stored_Avg_Flight_Time': features_avg_flight_time_per_user
})

print("Comparison of recalculated and stored average flight times per subject:")
print(comparison_df)



"""
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import pymysql  # MySQL connector
from src.config import Config

# Initialize the database connection
config = Config()
engine = config.get_engine()

# Query the database, using avg_flight_time
query = 
"""
#SELECT subject, sessionIndex, rep, avg_flight_time 
#FROM KeyLoggerFull.Features_Created;
"""
data = pd.read_sql(query, con=engine)

# Calculate the average flight time per user
average_total_flight_time_per_user = (
    data.groupby('subject')['avg_flight_time']
    .mean()
    .sort_values()
)

# Plot the graph with precision improvements
plt.figure(figsize=(12, 6))
bars = average_total_flight_time_per_user.plot(kind='bar', color='teal', width=0.8)

# Add data labels to the graph
for bar in bars.patches:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2, height + 0.02,  # Adjust spacing above the bar
        f'{height:.4f}', ha='center', va='bottom', fontsize=9, rotation=90,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')  # Background for clarity
    )

# Customize the graph
plt.xlabel('User ID')
plt.ylabel('Average Flight Time per Password Entry (Seconds)')
plt.title('Precise Average Flight Time for Full Password Entry per User')
plt.xticks(rotation=90)  # Rotate x-axis labels for clarity
plt.grid(axis='y', linestyle='--', alpha=.7)

# Display the graph
plt.show()

"""

"""
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset from the specified path
file_path = './data/DSL-StrongPasswordData.csv'
data = pd.read_csv(file_path)

# Identify columns related to flight times (UD columns)
ud_columns = [col for col in data.columns if 'UD' in col]

# Calculate the total flight time for each password entry
data['total_flight_time'] = data[ud_columns].sum(axis=1)

# Average flight time per user
average_total_flight_time_per_user = data.groupby('subject').mean()['total_flight_time']

# Sort values into appropriate orders
sorted_flight_times = average_total_flight_time_per_user.sort_values()

# Plot the graph with precision improvements
plt.figure(figsize=(12, 6))
bars = sorted_flight_times.plot(kind='bar', color='teal', width=0.8)

# Data Labels added to the graph
for bar in bars.patches:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2, height + 0.20,  # More space above the bar
        f'{height:.4f}', ha='center', va='bottom', fontsize=9, rotation=90,  # Rotation adjusted
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')  # Background box for clarity
    )

# Customize the graph
plt.xlabel('User ID')
plt.ylabel('Average Flight Time per Password Entry (Seconds)')
plt.title('Precise Average Flight Time for Full Password Entry per User')
plt.xticks(rotation=90)  # Adjust x-axis labels for clarity
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the graph
plt.show()


"""


"""
plt.figure(figsize=(12, 6))
data.boxplot(column='total_flight_time', by='subject', grid=False)
plt.xticks(rotation=90)
plt.xlabel('User ID')
plt.ylabel('Total Flight Time per Password Entry (Seconds)')
plt.title('Distribution of Flight Times per User')
plt.suptitle('')  # Removes the automatic title
plt.show()
"""


"""user_data = data[data['subject'] == 's002']  # Filter for a specific user
plt.figure(figsize=(10, 5))
plt.plot(user_data['total_flight_time'], marker='o')
plt.xlabel('Password Attempt')
plt.ylabel('Total Flight Time (Seconds)')
plt.title('Flight Time Across Password Attempts for User s002')
plt.show()
"""
