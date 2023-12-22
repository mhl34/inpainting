import torch
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset

class InpaintingDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.maskTransform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomErasing(p=1, scale=(0.1,0.1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img = self.dataset[index][0]
        img1 = self.maskTransform(img)
        img2 = self.transform(img)

        return (img1, img2)
    
    @staticmethod
    def to_pil_image(tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy().transpose((1, 2, 0))  # Channels last
            tensor = (tensor * 255).astype('uint8')
            return transforms.ToPILImage()(tensor)
        else:
            return tensor