
import json
import os, shutil

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import training_utils
import matplotlib.pyplot as plt
from recs_ecosystem_creator_rl.environment import environment
from recs_ecosystem_creator_rl.recommender import agent
from recs_ecosystem_creator_rl.recommender import data_utils
from recs_ecosystem_creator_rl.recommender import runner
from recs_ecosystem_creator_rl.recommender import value_model
from recs_ecosystem_creator_rl.recommender import ecoagent_experiment

def main():
    ecoagent_experiment.main()
