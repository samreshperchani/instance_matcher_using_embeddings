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
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

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

# path where extracted text will be stored
EXTRACTED_TEXT_DIR = config.EXTRACTED_TEXT_DIR

class OAEI:

    def generate_file_oaei_format(self, df):
        
        
        ns_map = {'':'http://knowledgeweb.semanticweb.org/heterogeneity/alignment','rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#', 'xsd': 'http://www.w3.org/2001/XMLSchema#'}

        for prefix, uri in ns_map.items():
            ET.register_namespace(prefix, uri)
        
        tree = ET.ElementTree()
        root = ET.Element("{" + ns_map['rdf'] + "}RDF")

        tree._setroot(root)

        alignment_tag = ET.Element("Alignment")
        
        xml_tag = ET.Element("xml")
        xml_tag.text= "1"
        alignment_tag.append(xml_tag)


        level_tag = ET.Element("level")
        level_tag.text= "0"
        alignment_tag.append(level_tag)
        
        
        type_tag = ET.Element("type")
        type_tag.text= "??"
        alignment_tag.append(type_tag)


        for index, row in df.iterrows():
            map_tag = ET.Element("map")
            cell_tag = ET.Element("Cell")

            entity1_tag = ET.Element("entity1")
            entity_1_id = str(row['entity_id_wiki_1'])
            
            if entity_1_id.startswith('<'):
                entity_1_id = entity_1_id[1:len(entity_1_id)]
            
            if entity_1_id.endswith('>'):
                entity_1_id = entity_1_id[0:len(entity_1_id)-1]
                
            entity1_tag.set("{" + ns_map['rdf'] + "}resource", entity_1_id)

            entity2_tag = ET.Element("entity2")
            entity_2_id = str(row['entity_id_wiki_1'])
            
            if entity_2_id.startswith('<'):
                entity_2_id = entity_2_id[1:len(entity_2_id)]
            
            if entity_2_id.endswith('>'):
                entity_2_id = entity_2_id[0:len(entity_2_id)-1]
            entity2_tag.set("{" + ns_map['rdf'] + "}resource", str(row['entity_id_wiki_2']))
            
            relation_tag = ET.Element("relation")
            
            if row['label'] == 1:
                relation_tag.text = "="
            else:
                relation_tag.text = "%"
            
            measure_tag = ET.Element("measure")
            measure_tag.text = "1.0"
            measure_tag.set("{" + ns_map['rdf'] + "}datatype", "xsd:float")


            map_tag.append(cell_tag)
            cell_tag.append(entity1_tag)
            cell_tag.append(entity2_tag)
            cell_tag.append(relation_tag)
            cell_tag.append(measure_tag)
            alignment_tag.append(map_tag)

        ET.register_namespace('',"http://knowledgeweb.semanticweb.org/heterogeneity/alignment")
        root.append(alignment_tag)
        #print(tree)
        #tree.write("sample.xml", encoding='utf-8',  xml_declaration=True, method = 'xml')
        return tree
