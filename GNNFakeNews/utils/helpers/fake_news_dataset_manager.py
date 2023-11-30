import json
import pickle
from os import path, listdir

from GNNFakeNews.utils.enums import GNNDatasetTypeEnum
from GNNFakeNews.utils.helpers.gnn_dataset_manager import DATA_DIR

DATASET_DIR = path.join(DATA_DIR, 'fakenewsnet_dataset')
LABELS = ['fake', 'real']
NEWS_CONTENT_FILE = 'news content.json'

POL_ID_TWITTER_MAPPING_PKL = 'pol_id_twitter_mapping.pkl'
GOS_ID_TWITTER_MAPPING_PKL = 'gos_id_twitter_mapping.pkl'

POL_ID_TIME_MAPPING_PKL = 'pol_id_time_mapping.pkl'
GOS_ID_TIME_MAPPING_PKL = 'gos_id_time_mapping.pkl'


class FakeNewsNetDatasetManager:
    def __init__(self, dataset_type: GNNDatasetTypeEnum, verbose=False):
        self.dataset_type = dataset_type
        self.label_id_dict = {}
        # existing ids in FakeNewsNet dataset
        self.news_existing_ids = []
        # news_id, index mapping, obtained from self.index_news_id_dict
        self.news_id_index_dict = {}
        # obtained from .pkl files.
        _, self.index_news_id_dict = get_news_id_node_id_user_id_dict(dataset_type)
        # initialize self.news_id_index_dict
        self._create_news_id_index_dict()
        # create the directory that given dataset_type resides in.
        self.directory = path.join(DATASET_DIR, dataset_type.value)
        # collect all ids in the FakeNewsNet dataset and initialize self.news_existing_ids
        self._collect_ids_in_dataset()

        if verbose:
            print(f'News existing ids: {self.news_existing_ids}')
            print(f'Label news id dict: {self.label_id_dict}')
            print(f'Index news id dict: {self.index_news_id_dict}')
            print(f'News id index dict: {self.news_id_index_dict}')

    def _create_news_id_index_dict(self):
        for index, news_id in self.index_news_id_dict.items():
            self.news_id_index_dict[news_id] = index

    def _collect_ids_in_dataset(self):
        for label in LABELS:
            self.label_id_dict[label] = []
            label_dir = path.join(self.directory, label)
            for folder in listdir(label_dir):
                self.label_id_dict[label].append(folder)
                # also collect the info about the news content being existent
                folder_dir = path.join(label_dir, folder)
                if NEWS_CONTENT_FILE in listdir(folder_dir):
                    self.news_existing_ids.append(folder)

    def get_existing_news_indexes_for_torch_dataset(self):
        ds_indexes = []
        missing_news_ids = []
        for news_id in self.news_existing_ids:
            try:
                ds_indexes.append(self.news_id_index_dict[news_id])
            except KeyError:
                missing_news_ids.append(news_id)
        return ds_indexes

    def get_entry_by_news_id(self, news_id: str):
        """
        get the entry by its id, e.g, gossipcop-2116458, returns a dict
        Parameters
        ----------
        news_id: str,
            the id of the news entry in the respective dataset
        """
        if news_id not in self.news_existing_ids:
            raise ValueError(f'This id: {news_id} does not have {NEWS_CONTENT_FILE}')
        for label in LABELS:
            label_dir = path.join(self.directory, label)
            if news_id in self.label_id_dict[label]:
                news_dir = path.join(label_dir, news_id)
                with open(path.join(news_dir, NEWS_CONTENT_FILE)) as f:
                    return json.load(f)

    def get_entry_by_torch_dataset_index(self, index: int):
        news_id = self.index_news_id_dict[index]
        return self.get_entry_by_news_id(news_id)


def load_pkl_files(dataset_type: GNNDatasetTypeEnum):
    """
    read and return the respective datasets pkl files.
    Parameters
    ----------
    dataset_type: the dataset  type of whose pkl files will be loaded
    """
    if dataset_type == GNNDatasetTypeEnum.POLITIFACT:
        node_id_user_id_file_name = POL_ID_TWITTER_MAPPING_PKL
        node_id_time_file_name = POL_ID_TIME_MAPPING_PKL
    else:
        node_id_user_id_file_name = GOS_ID_TWITTER_MAPPING_PKL
        node_id_time_file_name = GOS_ID_TIME_MAPPING_PKL
    with open(path.join(DATA_DIR, 'local', node_id_user_id_file_name), 'rb') as f:
        node_id_user_id_dict = pickle.load(f)
    with open(path.join(DATA_DIR, 'local', node_id_time_file_name), 'rb') as f:
        node_id_time_dict = pickle.load(f)
    return node_id_user_id_dict, node_id_time_dict


def get_news_id_node_id_user_id_dict(dataset_type: GNNDatasetTypeEnum):
    """
    read pkl files and return the mapping for each news file.
    Parameters
    ----------
    dataset_type: the dataset  type of whose pkl files will be loaded
    """
    node_id_user_id_dict, node_id_time_dict = load_pkl_files(dataset_type)
    structured_dict = {}
    index_news_id_dict = {}
    news_id = None
    news_index = 0
    for node_id, user_id in node_id_user_id_dict.items():
        try:
            user_id_int = int(user_id)
            structured_dict[news_id][actual_node_id] = user_id_int
            actual_node_id += 1
        except ValueError:
            news_id = user_id
            structured_dict[news_id] = {}
            # we start with 1 since 0 is reserved for news itself.
            actual_node_id = 1
            index_news_id_dict[news_index] = news_id
            news_index += 1

    return structured_dict, index_news_id_dict
