import os

# Base directory of the entire project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory definition
MODELS_DIR = os.path.join(BASE_DIR, 'app', 'models')
PREPROCESSING_DIR = os.path.join(BASE_DIR, 'app', 'preprocessing')
TRAINED_MODELS_DIR = os.path.join(MODELS_DIR, 'trained')
PREPROCESSING_PARAMS_DIR = os.path.join(PREPROCESSING_DIR, 'parameters')

TESTS_DIR = os.path.join(BASE_DIR, 'tests' )
TEST_EXAMPELS_DIR = os.path.join(TESTS_DIR, 'examples')