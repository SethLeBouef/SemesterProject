# Import necessary libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from src.config import Config


config = Config()
engine = config.get_engine()

# S Load Data from Keystroke_Raw Table
query = """
SELECT * FROM KeyLoggerFull.Keystroke_Raw
"""
keystroke_data = pd.read_sql_query(query, con=engine)

# Rename columns based on Keystroke_Raw structure
keystroke_data.columns = ['subject', 'sessionIndex', 'rep', 'H.period', 'DD.period.t', 'UD.period.t', 'H.t', 
                          'DD.t.i', 'UD.t.i', 'H.i', 'DD.i.e', 'UD.i.e', 'H.e', 'DD.e.five', 'UD.e.five', 
                          'H.five', 'DD.five.Shift.r', 'UD.five.Shift.r', 'H.Shift.r', 'DD.Shift.r.o', 
                          'UD.Shift.r.o', 'H.o', 'DD.o.a', 'UD.o.a', 'H.a', 'DD.a.n', 'UD.a.n', 'H.n', 
                          'DD.n.l', 'UD.n.l', 'H.l', 'DD.l.Return', 'UD.l.Return', 'H.Return']

#  Handle Missing Values
if keystroke_data.isnull().values.any():
    print("Missing values detected. Filling with column means...")
    keystroke_data.fillna(keystroke_data.mean(), inplace=True)

#  Calculate Features
# Basic Metrics
keystroke_data['total_characters'] = keystroke_data[['H.period', 'H.t', 'H.i', 'H.e', 'H.five', 
                                                     'H.Shift.r', 'H.o', 'H.a', 'H.n', 'H.l', 
                                                     'H.Return']].count(axis=1)

keystroke_data['avg_flight_time'] = keystroke_data[['DD.period.t', 'DD.t.i', 'DD.i.e', 'DD.e.five', 
                                                    'DD.five.Shift.r', 'DD.Shift.r.o', 'DD.o.a', 
                                                    'DD.a.n', 'DD.n.l', 'DD.l.Return']].mean(axis=1)

keystroke_data['avg_dwell_time'] = keystroke_data[['H.period', 'H.t', 'H.i', 'H.e', 'H.five', 
                                                   'H.Shift.r', 'H.o', 'H.a', 'H.n', 'H.l', 
                                                   'H.Return']].mean(axis=1)

keystroke_data['total_typing_duration'] = (
    keystroke_data['avg_flight_time'] * keystroke_data['total_characters'] + 
    keystroke_data['avg_dwell_time'] * keystroke_data['total_characters']
)

keystroke_data['hold_ratio'] = keystroke_data['avg_dwell_time'] / keystroke_data['total_typing_duration']
keystroke_data['CPS'] = keystroke_data['total_characters'] / keystroke_data['total_typing_duration']
keystroke_data['WPM'] = keystroke_data['CPS'] * 60 / 5  # Assuming an average word length of 5 characters

# Advanced Metrics
keystroke_data['std_flight_time'] = keystroke_data[['DD.period.t', 'DD.t.i', 'DD.i.e', 'DD.e.five', 
                                                    'DD.five.Shift.r', 'DD.Shift.r.o', 'DD.o.a', 
                                                    'DD.a.n', 'DD.n.l', 'DD.l.Return']].std(axis=1)

keystroke_data['std_dwell_time'] = keystroke_data[['H.period', 'H.t', 'H.i', 'H.e', 'H.five', 
                                                   'H.Shift.r', 'H.o', 'H.a', 'H.n', 'H.l', 
                                                   'H.Return']].std(axis=1)

keystroke_data['pause_ratio'] = keystroke_data['avg_flight_time'] / keystroke_data['avg_dwell_time']


keystroke_data['transition_frequency_t_i'] = (
    keystroke_data['H.i'] - keystroke_data['H.t']
).abs() / keystroke_data['total_typing_duration']

keystroke_data['inter_key_delay_std'] = keystroke_data[['DD.period.t', 'DD.t.i', 'DD.i.e']].std(axis=1)

# Session-Level Features
session_stats = keystroke_data.groupby(['subject', 'sessionIndex']).agg({
    'total_typing_duration': ['mean', 'std'],
    'avg_flight_time': ['mean', 'std'],
    'avg_dwell_time': ['mean', 'std'],
    'CPS': ['mean', 'std'],
    'WPM': ['mean', 'std'],
}).reset_index()

# Rename the aggregated columns for clarity
session_stats.columns = [
    'subject', 'sessionIndex',
    'total_typing_duration_mean', 'total_typing_duration_std',
    'avg_flight_time_mean', 'avg_flight_time_std',
    'avg_dwell_time_mean', 'avg_dwell_time_std',
    'CPS_mean', 'CPS_std',
    'WPM_mean', 'WPM_std'
]

# Merge session-level stats back into the main data
keystroke_data = pd.merge(keystroke_data, session_stats, how='left', on=['subject', 'sessionIndex'])

# Ratio-Based Features
keystroke_data['flight_to_dwell_ratio'] = keystroke_data['avg_flight_time'] / keystroke_data['avg_dwell_time']
keystroke_data['typing_duration_per_character'] = keystroke_data['total_typing_duration'] / keystroke_data['total_characters']

# Combined Features
keystroke_data['typing_efficiency'] = keystroke_data['WPM'] * keystroke_data['hold_ratio']

#  Save the Data with New Features
keystroke_data.to_sql('Features_Created', con=engine, if_exists='replace', index=False)
print("Feature creation complete, and data saved to Features_Created table.")