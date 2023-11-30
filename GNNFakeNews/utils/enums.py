from enum import Enum


class ExtendedEnum(Enum):
    @classmethod
    def all_elements(cls):
        return list(map(lambda c: c, cls))


class DeviceTypeEnum(ExtendedEnum):
    """
    enum class to handle static device types for GNN deprecated
    """
    CPU = 'cpu'
    GPU = 'cuda'
    # further specifications can be added


class GNNDatasetTypeEnum(ExtendedEnum):
    """
    enum class to handle static dataset types for GNN deprecated
    """
    POLITIFACT = 'politifact'
    GOSSIPCOP = 'gossipcop'


class GNNFeatureTypeEnum(ExtendedEnum):
    """
    enum class to handle static feature types for GNN deprecated
    """
    PROFILE = 'profile'
    SPACY = 'spacy'
    BERT = 'bert'
    CONTENT = 'content'


class GNNModelTypeEnum(ExtendedEnum):
    """
    enum class to handle different GNN model types
    """
    BIGCN = 'bigcn'
    VANILLA_GCNFN = 'vanilla_gcnfn'
    UPFD_GCNFN = 'upfd_gcnfn'
    GCN_GNN = 'gcn_gnn'
    GAT_GNN = 'gat_gnn'
    SAGE_GNN = 'sage_gnn'
    GNNCL = 'gnncl'
