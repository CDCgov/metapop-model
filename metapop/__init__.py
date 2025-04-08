# This file is required to make this directory (metapop) a package.
# It imports all the necessary modules and functions from the package
# and can also include initialization code if needed

# import in order of increasing dependencies to avoid circular imports - any two modules should not depend on each other
from .version import *
from .helper import *
from .model import * # depends on helper
from .app_helper import * # depends on helper, model
from .app import * # depends on helper, model, app_helper
from .app_with_table import * # depends on helper, model, app_helper

# when declared, this variable defines the public modules, subpackages and other named objects that should be imported when a user uses `from metapop import *`, otherwise, everything from the modules defined above will be imported
# __all__ = ["SEIRModel", "simulate", "get_percapita_contact_matrix"]
