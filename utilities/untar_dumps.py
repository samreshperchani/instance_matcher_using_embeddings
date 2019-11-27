import os
import glob
import tarfile
import multiprocessing as mp
import shutil
import sys
from pathlib import Path

path = Path(os.path.abspath(__file__))

# set configuration file path
#config_path = os.path.dirname(os.getcwd()) + '/config' 
config_path = str(path.parent.parent) + '/config' 

# add config file path to system
sys.path.append(config_path)

import config

# base directory of the solution
BASE_DIR = config.BASE_DIR

# path to data directory
DATA_DIR = config.DATA_DIR

# path where xml dumps are present in .tar format
DATA_DUMPS_DIR = config.DATA_DUMPS_DIR

# path where untared xml dumps to be placed
UNTARED_DUMPS_DIR = config.UNTARED_DUMPS_DIR

# path where processed dumps have to be placed
PROCESSED_DUMPS_DIR = config.PROCESSED_DUMPS_DIR


# target directory where dumps are to be extracted
TARGET_DIR_XML_DUMPS = BASE_DIR + '/' + DATA_DIR + '/' + UNTARED_DUMPS_DIR

# target directory where processed ttl file are to be extracted
TARGET_DIR_PROCESSED_DUMPS = BASE_DIR + '/' + DATA_DIR + '/' + PROCESSED_DUMPS_DIR

# extract dump
def extract_dump_file(dump_file):
        print('processing: ', dump_file)
        wiki_name = os.path.basename(dump_file).replace('.tar.gz', '')
        
        # xml dump path
        xml_dump_path = TARGET_DIR_XML_DUMPS + '/' + wiki_name
        
        # processed wiki path
        pr_wiki_path = TARGET_DIR_PROCESSED_DUMPS + '/' + wiki_name
        
        if (os.path.exists(xml_dump_path)):
                shutil.rmtree(xml_dump_path)

        if (os.path.exists(pr_wiki_path)):
                shutil.rmtree(pr_wiki_path)
                
                
        os.mkdir(xml_dump_path)
        os.mkdir(pr_wiki_path)

        try:
                with tarfile.open(dump_file, "r|*") as tar:
                        for member in tar:
                                if member.isfile():
                                        # check if xml file is there then extract it
                                        if member.name.endswith('.xml'):
                                                tar.extract(member, path = xml_dump_path)
                                        elif member.name.endswith('.ttl'):
                                                tar.extract(member, path = pr_wiki_path)

        except Exception as e:
                print(e)


# function to call extract
def untared_xml_dumps():
        # get list of all dumps
        FILES_LIST = glob.glob(BASE_DIR + '/' + DATA_DIR + '/' + DATA_DUMPS_DIR + '/*')

        # get some processes
        processes = max(1, mp.cpu_count()-1)
        
        # run extraction in parallel
        with mp.Pool(processes) as pool:
                pool.map_async(extract_dump_file, FILES_LIST)
                pool.close()
                pool.join()


# main function which will be called when script will be executed
if __name__ == '__main__':
        # call function to extract dumps
        if not os.path.exists(TARGET_DIR_XML_DUMPS):
                os.mkdir(TARGET_DIR_XML_DUMPS)
                
        if not os.path.exists(TARGET_DIR_PROCESSED_DUMPS):
                os.mkdir(TARGET_DIR_PROCESSED_DUMPS)
        untared_xml_dumps()