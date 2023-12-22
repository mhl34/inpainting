import torch

class hyperParams:
    def __init__(self):
        self.epochs = 20
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.batch_size = 128
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'