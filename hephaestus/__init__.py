# ruff: noqa: F401, F403
"""Hephaestus package initialization."""

# Import necessary modules for accessibility
from hephaestus.training.training_loop import train_model

# Since there are circular imports, we need to be careful about the order
# First import utils, which doesn't depend on other modules
from .utils import *

# Then import models, which may depend on utils
from .models import *

# Finally import analysis and training, which depend on models
from .analysis import *
from .training import *
