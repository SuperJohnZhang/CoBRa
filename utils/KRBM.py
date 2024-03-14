# The main entry for the generation of all VQAs in the benchmark
import argparse
import torch
import numpy as np
import diffusers
from kops.element import Element
import os
import pandas as pd
from utils.io import WITIO
import logging
from kops.element import KNet
import kops.inference

logger = logging.getLogger(name="KRBM")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

args = argparse.ArgumentParser()
args.add_argument("--witdir", default='../../wit/sample/', \
                  help='the folder containing wit annotations')
args.add_argument("--datadir", default='./data/', help='the folder to hold generated data')
args.add_argument("--restore", action='store_true', \
                  help='whether restore from some preprocessed csv')
args = args.parse_args()

# print of parameters
for k,v in vars(args).items():
    logger.info('{}: {}'.format(k, v))

# create folder for output if not exist
rimg_path = os.path.join(args.datadir, 'rimg')
img_path = os.path.join(args.datadir, 'img')
QA_path = os.path.join(args.datadir, 'QA') # the intermediate wit ann/graph is put in the same folder as QA
paths = {'rimg': rimg_path, 
         'img': img_path, 
         'QA': QA_path}
for path in paths.values():
    if not os.path.isdir(path):
        os.makedirs(path)

wit_paths = [os.path.join(args.witdir, filename) for filename in os.listdir(args.witdir)]
logger.info('-'*20)
logger.info('new wit annotations to load')
logger.info(wit_paths)
for path in wit_paths:
    witdf = pd.read_csv(path, delimiter='\t', header=0)

cols = ['image_url', 'caption_attribution_description', 'context_page_description']
wit = WITIO(witdf, cols=cols, datapaths=paths, 
            restore=args.restore, verbose=True)
wit._download_all()
# wit._download_by_index(10000)
wit._save_to_csv()

#