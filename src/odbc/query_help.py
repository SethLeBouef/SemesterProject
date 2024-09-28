import pandas as pd
from src.odbc.conn_manager import create_connection

def insertSQLPandas(df, table, uri='mysql', if_exists='append', index=False, chunksize=10000, method='multi', database=None):
    """Insert a DataFrame into a SQL table."""
    if database is not None:
        with create_connection(database=database, uri=uri) as connection:
            #df.to_sql(table, connection, if_exists=if_exists, index=index, chunksize=chunksize, method=method)
            df.to_sql(table, connection, if_exists=if_exists, index=index)
    else:
        with create_connection(uri=uri) as connection:
            #df.to_sql(table, connection, if_exists=if_exists, index=index, chunksize=chunksize, method=method)
            df.to_sql(table, connection, if_exists=if_exists, index=index)

def selectSQLPandas(sql, database=None, uri='mysql'):
    """Execute a SQL query and return the results as a DataFrame."""
    if database is not None:
        with create_connection(database=database, uri=uri) as connection:
            results = pd.read_sql(sql, connection)
    else:
        with create_connection(uri=uri) as connection:
            results = pd.read_sql(sql, connection)
    return results


