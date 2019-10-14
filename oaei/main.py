from oaei import OAEI
import glob
import logging
import datetime
import os
import re
import gensim
import sys
from pathlib import Path
import shutil
import multiprocessing as mp
import pandas as pd
import random
import time
import sql
from lxml import etree

logger = logging.getLogger(__name__)

# set configuration file path
config_path = os.path.dirname(os.getcwd()) + '/config' 

# add config file path to system
sys.path.append(config_path)

import config

# base directory of the solution
BASE_DIR = config.BASE_DIR

# path to data directory
DATA_DIR = config.DATA_DIR

# path where extracted text will be stored
EXTRACTED_TEXT_DIR = config.EXTRACTED_TEXT_DIR


df = pd.read_pickle(BASE_DIR  + '/' + DATA_DIR + '/ensemble_dataset/test_set/test_set.pkl')

oaei = OAEI()

print(df.head())
oaei.generate_file_oaei_format(df.head())
