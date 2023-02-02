# %% main file, i.e., runner

#%config InlineBackend.figure_formats = ['png','pdf']
#import sys,os
#sys.path.append('..')

from scenarios import lora_scenario as s
from core.TrainingManager import TrainingManager as TM
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
import numpy as np
import timeit
import sys


#matplotlib_inline.backend_inline.set_matplotlib_formats('png','pdf')
#set_matplotlib_formats('png', 'pdf')

TRAIN = False
TEST = False
upperbound = 100

## Check if the type of execution is provided (test or training)
def check_args():
    global TRAIN
    global TEST
    global upperbound


    if len(sys.argv) == 3:
        if sys.argv[1] == "train" :
            TRAIN = True
            TEST = False
            upperbound = sys.argv[2]
        else:
            if sys.argv[1] == "test" :
                TRAIN = False
                TEST = True
            else:
                print("Error! You need to provide an input argument for selecting between \"run.py test\" and \"run.py train\"")
                sys.exit()
            
    else:
        print("Error! You need to provide an input argument for selecting between \"run.py test\" and \"run.py train\"")
        sys.exit()


#TRAIN = True
#TEST = False
#TRAIN = False
#TEST = True
def main():
    
    check_args()
    
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
        tm.run(upperbound,verbose=True)
        end = timeit.default_timer()
        print('\n It took ~{} useconds'.format(str(round(end-start))))

    if TEST:
        start = timeit.default_timer()
        # let it do the magic
        tm.test(upperbound,verbose=True)
        end = timeit.default_timer()
        print('\n It took ~{} useconds'.format(str(round(end-start))))

if __name__ == "__main__":
    main()

