import abc
import os
from dotenv import load_dotenv

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
    """Configuration class to store the state of bools for different scripts access."""

    def __init__(self):
        """Initialize the Config class."""
        self.mssql_server = os.getenv("MSSQL_SERVER")
        self.mysql_server = os.getenv("MYSQL_SERVER")
        self.mssql_driver = os.getenv("MSSQL_DRIVER")
        self.mssql_uid = os.getenv("MSSQL_UID")
        self.mssql_pid = os.getenv("MSSQL_PID")
        self.mysql_uid = os.getenv("MYSQL_UID")
        self.mysql_pid = os.getenv("MYSQL_PID")
        self.mssql_db = os.getenv("MSSQL_DB")
        self.mysql_db = os.getenv("MYSQL_DB")

    def __repr__(self):
        """Representation of the Config class."""
        attrs = ",\n".join(
            f" {attr}={repr(getattr(self, attr))}"
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        )
        return f"Config(\n{attrs}\n)"

