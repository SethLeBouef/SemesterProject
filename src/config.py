import abc
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pymysql  # or any other required MySQL library

# Load environment variables from .env file
load_dotenv()

class Singleton(abc.ABCMeta, type):
    """Singleton metaclass for ensuring only one instance of a class."""
    
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class AbstractSingleton(abc.ABC, metaclass=Singleton):
    """Abstract singleton class used for future singleton classes."""
    pass

class Config(metaclass=Singleton):
    """Configuration singleton to manage environment variables and database connection."""

    def __init__(self):
        """Initialize the Config singleton class with environment variables."""
        # Load database credentials from environment variables
        self.mysql_server = os.getenv("MYSQL_SERVER", "localhost")
        self.mysql_uid = os.getenv("MYSQL_UID")
        self.mysql_pid = os.getenv("MYSQL_PID")
        self.mysql_db = os.getenv("MYSQL_DB")
        
        # Set up the connection URL and SQLAlchemy engine
        self.connection_url = (
            f"mysql+pymysql://{self.mysql_uid}:{self.mysql_pid}"
            f"@{self.mysql_server}/{self.mysql_db}"
        )
        self.engine = create_engine(self.connection_url)

    def get_engine(self):
        """Return the SQLAlchemy engine."""
        return self.engine

    def get_connection(self):
        """Return a database connection from the Config class."""
        return Config().get_connection()

    def __repr__(self):
        """Representation of the Config class."""
        attrs = ",\n".join(
            f" {attr}={repr(getattr(self, attr))}"
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        )
        return f"Config(\n{attrs}\n)"