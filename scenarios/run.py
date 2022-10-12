# %% main file, i.e., runner

import sys,os
sys.path.append('..')

from .scenarios import lora_scenario as s
from .core.TrainingManager import TrainingManager as TM

import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
import numpy as np

import timeit
set_matplotlib_formats('png', 'pdf')
TRAIN = True
def main():
    # define a training manager object
    tm = TM(s.num_episodes, 
            s.episode_length, 
            s.agent,
            s.env,
            log_file=s.log_file)

    print('Scenario:%s' % s.title)
    if TRAIN:
        start = timeit.default_timer()
        # let it do the magic
        tm.run(verbose=False)
        end = timeit.default_timer()
        print('\n It took ~{} useconds'.format(str(round(end-start))))