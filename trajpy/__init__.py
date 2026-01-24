try:
    from importlib.metadata import version
except ImportError:
    version = None

__version__ = version("trajpy")
name = "trajpy"
