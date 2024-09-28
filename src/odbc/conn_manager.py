import urllib.parse
from sqlalchemy import create_engine
from src.config import Config
config = Config()

def create_connection(uri='mysql', database=None):
    """Create a connection string."""
    if uri == 'mssql':
        if database is None:
            database = config.mssql_db
        db_uri = (
            f"mssql+pyodbc://{config.mssql_uid}:"
            f"{urllib.parse.quote_plus(config.mssql_pid)}@{config.mssql_server}/"
            f"{database}?driver={urllib.parse.quote_plus(config.mssql_driver)}&Encrypt=no"
        )
    elif uri == 'mysql':
        if database is None:
            database = config.mysql_db
        db_uri = (
            f"mysql+pymysql://{config.mysql_uid}:"
            f"{config.mysql_pid}@{config.mysql_server}/{database}"
        )
    else:
        raise ValueError("Unsupported URI scheme. Please use 'mssql' or 'mysql'.")

    engine = create_engine(db_uri)
    return engine.begin()
