# This file is required to make this directory (metapop) a package.
# It imports all the necessary modules and functions from the package
# and can also include initialization code if needed


# from each module, define and import all the functions shared between modules
# any two modules should not depend on each other. this results in a circular
# import and will produce errors. for clarity, we name the dependencies of each
# module in the comments here beside the module import
from .advanced_app import advanced_app  # depends on app_helper
from .app import app  # depends on app_helper
from .app_helper import (
    get_scenario_results,
    read_parameters,
)  # depends on sim

# from .app_with_table import app_with_table  # depends on  app_helper
from .helper import (
    Ind,
    build_vax_schedule,
    calculate_beta_factor,
    calculate_foi,
    construct_beta,
    get_infected,
    get_metapop_info,
    get_percapita_contact_matrix,
    get_r0,
    get_r0_one_group,
    initialize_population,
    rate_to_frac,
    rescale_beta_matrix,
    time_to_rate,
    vaccinate_groups,
)  # depends on version
from .model import (
    SEIRModel,
)  # depends on helper
from .sim import run_model, simulate, simulate_replicates  # depends on model
from .version import __git_commit__, __version__, __versiondate__  # depends on nothing

# when declared, this variable defines the public modules, subpackages and
# other named objects that should be available when a user uses
# `import metapop as mt`

__all__ = [
    "get_percapita_contact_matrix",
    "get_r0",
    "rescale_beta_matrix",
    "calculate_beta_factor",
    "get_r0_one_group",
    "construct_beta",
    "initialize_population",
    "get_infected",
    "calculate_foi",
    "rate_to_frac",
    "time_to_rate",
    "build_vax_schedule",
    "vaccinate_groups",
    "Ind",
    "SEIRModel",
    "run_model",
    "simulate",
    "simulate_replicates",
    "get_scenario_results",
    "read_parameters",
    "app",
    "advanced_app",
    # "app_with_table",
    "get_metapop_info",
    "__version__",
    "__versiondate__",
    "__git_commit__",
]
