import math
import torch_geometric.transforms as T

from GNNFakeNews.utils.data_loader import DropEdge, ToUndirected
from GNNFakeNews.utils.enums import GNNModelTypeEnum, GNNDatasetTypeEnum, GNNFeatureTypeEnum


class HparamFactory:
    """
    factory class that generates different hparams for different deprecated
    """
    model_type = None
    dataset = None
    batch_size = None
    lr = None
    weight_decay = None
    n_hidden = None
    dropout_rates = None
    epochs = None
    feature = None
    transform = None
    pre_transform = None
    concat = None
    max_nodes = None

    def __init__(self, model_type: GNNModelTypeEnum, test_mode=False, **kwargs):
        self.model_type = model_type
        self._load_for_model(model_type)
        if test_mode:
            self._set_epochs_for_test()

        for key in self.__dict__.keys():
            if key in kwargs.keys():
                value = kwargs.pop(key, None)
                setattr(self, key, value)

        if self.dataset == GNNDatasetTypeEnum.GOSSIPCOP and model_type == GNNModelTypeEnum.GNNCL:
            self.max_nodes = 200

        print('#################################')
        print('-----> The hyperparameters are set!')
        for key in self.__dict__.keys():
            print(f'{key} = {getattr(self, key)}')
        print('#################################')

    def _set_epochs_for_test(self):
        self.epochs = math.ceil(self.epochs / 5)

    def _load_for_model(self, model_type: GNNModelTypeEnum):
        """
        Given a model type, method returns the initialized class instance
        """
        if model_type == GNNModelTypeEnum.BIGCN:
            self._load_for_bigcn()
        elif model_type == GNNModelTypeEnum.VANILLA_GCNFN:
            self._load_for_vanilla_gcnfn()
        elif model_type == GNNModelTypeEnum.UPFD_GCNFN:
            self._load_for_upfd_gcnfn()
        elif model_type in [GNNModelTypeEnum.GCN_GNN, GNNModelTypeEnum.GAT_GNN, GNNModelTypeEnum.SAGE_GNN]:
            self._load_for_gnn()
        elif model_type == GNNModelTypeEnum.GNNCL:
            self._load_for_gnncl()
        else:
            raise ValueError(f'Possible values are {GNNModelTypeEnum.all_elements()}')

    def _load_for_bigcn(self):
        self.dataset = GNNDatasetTypeEnum.POLITIFACT
        self.batch_size = 128
        self.lr = 0.01
        self.weight_decay = 0.001
        self.n_hidden = 128
        self.dropout_rates = {
            'TDdroprate': 0.2,
            'BUdroprate': 0.2,
        }
        self.epochs = 45
        self.feature = GNNFeatureTypeEnum.PROFILE
        self.transform = DropEdge(self.dropout_rates['TDdroprate'], self.dropout_rates['BUdroprate'])

    def _load_for_upfd_gcnfn(self):
        self._load_for_gcnfn()
        self.feature = GNNFeatureTypeEnum.SPACY
        self.concat = True

    def _load_for_vanilla_gcnfn(self):
        self._load_for_gcnfn()
        self.feature = GNNFeatureTypeEnum.CONTENT
        self.concat = False

    def _load_for_gcnfn(self):
        self.dataset = GNNDatasetTypeEnum.POLITIFACT
        self.batch_size = 128
        self.lr = 0.01
        self.weight_decay = 0.001
        self.n_hidden = 128
        self.epochs = 60
        self.transform = ToUndirected()

    def _load_for_gnn(self):
        self.dataset = GNNDatasetTypeEnum.POLITIFACT
        self.batch_size = 128
        self.lr = 0.01
        self.weight_decay = 0.01
        self.n_hidden = 128
        self.epochs = 35
        self.feature = GNNFeatureTypeEnum.BERT
        self.concat = True
        self.transform = ToUndirected()

    def _load_for_gnncl(self):
        self.dataset = GNNDatasetTypeEnum.POLITIFACT
        self.batch_size = 128
        self.lr = 0.001
        # self.weight_decay = 0.001 # set in args but never used in the original implementation
        self.n_hidden = 128
        self.epochs = 60
        self.feature = GNNFeatureTypeEnum.PROFILE
        self.max_nodes = 500
        self.transform = T.ToDense(self.max_nodes)
        self.pre_transform = ToUndirected()
