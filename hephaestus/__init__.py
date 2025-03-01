# ruff: noqa: F401, F403
"""Hephaestus package initialization."""

# Import necessary modules for accessibility
from hephaestus.training.train_utils import train_model

# Since there are circular imports, we need to be careful about the order
# First import utils, which doesn't depend on other modules
# Finally import analysis and training, which depend on models
from .analysis import *

# Then import models, which may depend on utils
from .models import *
from .training import *
from .utils import *

# Add the new training function
from hephaestus.training.train_model import train_model
