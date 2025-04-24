# This file is required to make this directory (metapop) a package.
# It imports all the necessary modules and functions from the package
# and can also include initialization code if needed

# import in order of increasing dependencies to avoid circular imports - any two modules should not depend on each other
# from each module, define and import all the functions shared between modules
from .version import *  # depends on nothing
from .helper import (
    get_percapita_contact_matrix,
    get_r0,
    rescale_beta_matrix,
    calculate_beta_factor,
    get_r0_one_group,
    construct_beta,
    initialize_population,
    get_infected,
    calculate_foi,
    rate_to_frac,
    time_to_rate,
    build_vax_schedule,
    vaccinate_groups,
)  # depends on nothing
from .model import (
    Ind,
    SEIRModel,
)  # depends on helper
from .sim import run_model, simulate  # depends on model
from .app_helper import (
    get_scenario_results,
    read_parameters,
)  # depends on sim
from .app import app  # depends on app_helper
from .advanced_app import advanced_app  # depends on app_helper
from .app_with_table import app_with_table  # depends on  app_helper


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
    "get_scenario_results",
    "read_parameters",
    "app",
    "advanced_app",
    "app_with_table",
]
