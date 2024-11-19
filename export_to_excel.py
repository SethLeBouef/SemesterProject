import pandas as pd
from sqlalchemy import create_engine

# Connect to the MySQL database
engine = create_engine("mysql+pymysql://root:7751:Zeus@localhost:3306/KeyloggerFull")

# Query to load the data from MySQL
query = "SELECT * FROM keystrokedata;"

# Load the data into a DataFrame
df = pd.read_sql(query, con=engine)

# Save the data to an Excel file
df.to_excel("keystrokedata_updated.xlsx", index=False)

print("Data successfully saved to 'keystrokedata_updated.xlsx'")