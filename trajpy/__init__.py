import pkg_resources

try:
    __version__ = pkg_resources.get_distribution('trajpy').version
except pkg_resources.DistributionNotFound:
    __version__ = ''