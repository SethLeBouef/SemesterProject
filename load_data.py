import pandas as pd 
from src.odbc import selectSQLPandas, insertSQLPandas

df = pd.read_excel('./data/ceo_data.xlsx')

insertSQLPandas(df,'ceo_data','mysql',database='KeyLogger')

