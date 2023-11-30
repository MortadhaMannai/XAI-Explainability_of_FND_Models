import torch

from GNNFakeNews.utils.enums import DeviceTypeEnum


class ModelArguments:
    """
    class to save the defaults and initial setup for PyTorch
    """
    seed = None
    device = None
    multi_gpu = None

    def __init__(self, seed=777, device=DeviceTypeEnum.GPU, multi_gpu=False):
        self.seed = seed
        if not torch.cuda.is_available() and device == DeviceTypeEnum.GPU:
            raise ValueError(f'device cannot be {DeviceTypeEnum.GPU}, because CUDA is not available.')
        self.device = torch.device(device.value)
        self.multi_gpu = multi_gpu
        self.setup()

    def setup(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
