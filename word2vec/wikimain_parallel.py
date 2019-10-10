import logging
import os
import re
import multiprocessing
from parsewikitextown import get_raw_text_and_links_from_markup
from processwikidump import extract_pages
from wikitokenize import tokenize_spacy
import tarfile
import ntpath
import shutil
import multiprocessing.pool
import time
import glob
import traceback
from random import randint
from gensim import utils
import sys
from pathlib import Path


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

# path where untared xml dumps are present
UNTARED_DUMPS_DIR = config.UNTARED_DUMPS_DIR


'''
# this class is to create parallel threads on top of parallel threads because text from XMLs are being extracted parallely
# and text for each individual XML is also being extracted in parallel way
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

'''
logger = logging.getLogger(__name__)
package_directory = os.path.dirname(os.path.abspath(__file__))


def process_link_mapping(title, links, wiki_name):
    processed_map = dict()
    
    # append wiki name in URIs
    for link_text, link_target in links.items():
        processed_map[link_text.strip().lower()] = 'http://dbkwik.webdatacommons.org/'.strip() + wiki_name.strip().lower() + '.wikia.com/resource/' + link_target.strip().replace(' ', '_').lower() + ' '
    # at the end of the url a whitespace to ensure the url ends there
    processed_map[title.strip().lower()] = 'http://dbkwik.webdatacommons.org/'.strip() + wiki_name.strip().lower() + '.wikia.com/resource/' + title.strip().replace(' ', '_').lower() + ' '

    replacement_list = sorted(list(processed_map.items()), key=lambda x: len(x[0]), reverse=True)
    return replacement_list

def replace_text(text, links, title, wiki_name):
    text = text.lower()
    replacement_list = process_link_mapping(title, links, wiki_name)
    for link_text, replacement in replacement_list:
        # a lookahead because we dont want to replace "abcdef" when we only have "abc" replace with "123"
        #(?<!\w)dor(?!\w)
        #replacement = replacement.replace('/','')
        replacement = replacement.replace('\\','')
        text = re.sub('(?<!\w)' + re.escape(link_text) + '(?!\w)', replacement, text)
    return text

def process_page(args):#title, text, pageid):
    title, text, pageid, wiki_name = args
    #print("Process " + title + ' (' + pageid + ')')
    text, links = get_raw_text_and_links_from_markup(text)
    text = replace_text(text, links, title, wiki_name)
    text = re.sub('\s+', ' ', text).strip()
    sentences = tokenize_spacy(text)
    return sentences, title, pageid

def process_wiki_dump(source, target, wiki_name, processes=None):
    if processes is None:
        processes = max(1, multiprocessing.cpu_count() - 1)
    #print(processes)

    with open(source, 'r', encoding='utf-8') as dump_file, \
         open(target, 'w', encoding='utf-8') as out_file:

        page_generator = extract_pages(dump_file,wiki_name, filter_namespaces=set(['0']))

        #for title, text, pageid in page_generator:
        #    sentences, title, pageid = process_page(title, text, pageid)
        #    for sentence in sentences:
        #        out_file.write(sentence + '\n')

        with multiprocessing.Pool(processes) as pool:
            for group in utils.chunkize(page_generator, chunksize=10 * processes, maxsize=1):
                for sentences, title, pageid in pool.imap(process_page, group):
                    for sentence in sentences:
                        out_file.write(sentence + '\n')


def extract_text_from_xml(file_path):
    target_file = ''
    try:
        xml_file_folder = file_path

        # get language code from xml folder name
        #language_code = xml_file_folder.split('~')[1]
        
        # have this check if only extraction of english wiki is required else convert it into if True:
        if True:

            # get list xml files in wiki folder
            xml_files = glob.glob(BASE_DIR + '/' + DATA_DIR + '/' + UNTARED_DUMPS_DIR + '/' + file_path + '/*.xml')
            
            # if xml file is present proceed with text extraction of this wiki
            if len(xml_files) > 0:
                xml_file = xml_files[0]
                file_name = ntpath.basename(xml_file)
                index_wikia_com = file_name.find('wikiacom')
                wiki_name = file_name[:index_wikia_com]
                target_file = xml_file_folder + '.txt'


                # if text is already present then do not extract agaian
                if not os.path.exists(BASE_DIR + '/' + DATA_DIR + '/' + EXTRACTED_TEXT_DIR + '/'  + target_file):
                    print('Processing: ', file_path)
                    process_wiki_dump(xml_file, BASE_DIR + '/' + DATA_DIR + '/' + EXTRACTED_TEXT_DIR + '/'  + target_file, wiki_name)
                else:
                    print(xml_file_folder, ' : text extraction already present for this wiki')
                    return
                
                print(xml_file_folder, ' : text extraction successful for this wiki')
            else:
                print(xml_file_folder, ' : XML file not present for this wiki')
    
    except Exception as e: 
        print(xml_file_folder,' : Exception occured ', e)
        traceback.print_exc()

        # in case text extraction is already present then delete the text file so that in next try text can be extracted again
        if target_file.strip() != '':
            if os.path.exists(BASE_DIR + '/' + DATA_DIR + '/' + EXTRACTED_TEXT_DIR + '/'  + target_file + '/' + target_file):
                os.remove(BASE_DIR + '/' + DATA_DIR + '/' + EXTRACTED_TEXT_DIR + '/'  + target_file + '/' + target_file) 


def extract_text_for_all_wikis():

    # get list of all folders in untared dumps directory
    file_names = os.listdir(BASE_DIR + '/' + DATA_DIR + '/' + UNTARED_DUMPS_DIR + '/')

    for file in file_names:
        extract_text_from_xml(file)

    # create some processes
    #processes = max(1, 20)
    
    # extract text parallely
    #with MyPool(processes) as pool:
    #    pool.map_async(extract_text_from_xml, file_names)
    #    pool.close()
    #    pool.join()


if __name__ == '__main__':
    
    if not os.path.exists(BASE_DIR + '/' + DATA_DIR + '/' +EXTRACTED_TEXT_DIR):
        os.mkdir(BASE_DIR + '/' + DATA_DIR + '/' +EXTRACTED_TEXT_DIR)
    
    extract_text_for_all_wikis()
