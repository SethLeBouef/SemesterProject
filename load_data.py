"""import pandas as pd 
from src.odbc import selectSQLPandas, insertSQLPandas

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('./data/DSL-StrongPasswordData.csv')


insertSQLPandas(df,'keystrokedataset','mysql',database='KeyloggerFull')
"""


###Code below is working on both the SQL database side with the KeyLoggerML DB with the KeyloggerFULL Schema
import pandas as pd
from sqlalchemy import create_engine

connection_url = "mysql+pymysql://root:7751:Zeus@localhost/KeyloggerFull"

engine = create_engine(connection_url)


try:
    df = pd.read_sql('SELECT * FROM keystrokedata', engine)
    print("Data successfully loaded:")
    print(df.head())
except Exception as e:
    print(f"An error occured: {e}")
    
 