import requests
import PIL
from PIL import Image
from io import BytesIO
import pandas as pd
from typing import List
import logging
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

logger = logging.getLogger(name="IO")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)


def download_image(url) -> Image:
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)',
    }
    response = session.get(url, headers=headers)
    return Image.open(BytesIO(response.content)).convert('RGB')
 

class WITIO(object):
    def __init__(self, wit:pd.DataFrame, \
                datapaths:dict, \
                cols:List[str]=['image_url', 'caption_attribution_description'],
                restore:bool=True,
                verbose:bool=False) -> None:
        
        self.paths = datapaths # include three paths rimg, img, QA
        self.cols = cols
        self.verbose = verbose
        if restore:
            self.data = pd.read_csv( \
                os.path.join(datapaths['QA'], 'wit_ann.csv'), \
                    index_col=0, header=0)
            if self.verbose:
                self._display()
            return

        # if no restore, start from a wit pd 
        self.data:pd.DataFrame = wit
        self._preprocess()
        pass

    # [todo] finish multi wit addition later
    # def add_wit(self, wit:pd.DataFrame):
    #     wit = wit.loc[wit['language']=='en']
    #     for col in self.cols:
    #         wit = wit.loc[wit[col].notna()]
    #     wit = wit.loc[wit['is_main_image']==True]
    #     wit.reset_index(drop=True, inplace=True)
    #     wit.insert(0, 'image_name', ['{}.jpg'.format(\
    #         rd) for rd in wit.index]) 
    #     wit.insert(1, 'downloaded', False)

    #     self._check_download_status(self.paths['rimg'])
    #     self._display()
    #     return 

    def _preprocess(self):
        # obtain the english content only
        self.data = self.data.loc[self.data['language']=='en']

        # remove examples with non
        for col in self.cols:
            self.data = self.data.loc[self.data[col].notna()]
        
        # choose only the main image
        self.data = self.data.loc[self.data['is_main_image']==True]

        # reset the index to drop original index with dropped rows
        self.data.reset_index(drop=True, inplace=True)

        # add a column to denote image name
        self.data.insert(0, 'image_name', ['{}.jpg'.format(\
            rd) for rd in self.data.index]) 

        # add a column to denote its download status
        self.data.insert(1, 'downloaded', False)

        self._check_download_status(self.paths['rimg'])
        if self.verbose:
            self._display()

    def _len(self):
        return self.data.shape[0]

    def _display(self):
        logger.info('# of samples available: {}, # of images downloaded: {}'.format( \
            self._len(), self.data['downloaded'].sum()))
    
    def _check_download_status(self, path:str):
        filenames = os.listdir(path)
        for filename in filenames:
            tmp = self.data.loc[self.data['image_name']==filename].index
            if tmp.shape[0] == 1:
                self.data.loc[tmp[0], 'downloaded'] = True
    
    def _download_by_index(self, i:int, pause:int=0):
        # if has already being download, skip
        if self.data.loc[i, 'downloaded']:
            return

        path = self.paths['rimg']
        img = download_image(self.data.loc[i, 'image_url'])
        img.save(os.path.join(path, self.data.loc[i, 'image_name']))
        self.data.loc[i, 'downloaded'] = True
        time.sleep(pause)
    
    def _download_all(self):
        for index, row in self.data.iterrows():
            try:
                self._download_by_index(index, 1)
            except PIL.UnidentifiedImageError as e:
                logger.info(row['image_url'])
            finally:
                pass

    def _save_to_csv(self):
        self.data.to_csv(os.path.join(self.paths['QA'], 'wit_ann.csv'))

    def __str__(self) -> str:
        return '# of samples: {}, # of images downloaded: {}'.format( \
            self._len(), self.data['downloaded'].sum())

    def __getitem__(self, index):
        return self.data.iloc[index]


class CoBRa(WITIO):
    def __init__(self, wit: pd.DataFrame, 
                 datapaths: dict, 
                 cols: List[str] = ['image_url', 'caption_attribution_description'], \
                 restore: bool = True, \
                 verbose: bool = False) -> None:
        super().__init__(wit, datapaths, cols, restore, verbose)

    def _build_KG(self):
        return
    
    def __call__(self, *args: Image.Any, **kwds: Image.Any) -> Image.Any:
        return super().__call__(*args, **kwds)
    
    def __str__(self) -> str:
        return super().__str__()