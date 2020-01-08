import os

# setting based on the winning solution
# time window in seconds
ANALYSIS_TIME_START = 0.5
ANALYSIS_TIME_END = 4.5
# frequency window in Hz
ANALYSIS_FREQUENCY_START = 12
ANALYSIS_FREQUENCY_END = 14


# path to data directory
DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

# path to results directory
RESULTS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'results')
