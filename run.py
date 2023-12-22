import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from ContextEncoder import ContextEncoder
from hyperParams import hyperParams
from InpaintingDataset import InpaintingDataset
from Loss import Loss
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt

class runModel():
    def __init__(self):
        pass

    def train(self, model, hyperparams, dataloader, criterion, optimizer):
        model.train()
        lossLst = []
        for epoch in range(hyperparams.epochs):
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}', unit='batch')
            for batch_idx, (img1, img2) in progress_bar:
                img1, img2 = img1.to(hyperparams.device), img2.to(hyperparams.device)

                preds = model(img1)

                loss = criterion(preds, img2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lossLst.append(loss.item())

                if batch_idx == 1:
                    img1_arr = torch.clamp(img1, 0, 1)[0].permute(1, 2, 0).detach().cpu().numpy()
                    img2_arr = torch.clamp(img2, 0, 1)[0].permute(1, 2, 0).detach().cpu().numpy()
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(img1_arr)
                    plt.title('Image 1')
                    plt.subplot(1, 2, 2)
                    plt.imshow(img2_arr)
                    plt.title('Image 2')
                    plt.show()

            print(f"Epoch {epoch + 1}   Average Training Loss: {sum(lossLst)/len(lossLst)}")
                

    def evaluate(self):
        pass

    def run(self):
        model = ContextEncoder()
        hp = hyperParams()
        criterion = Loss()
        optimizer = optim.Adam(model.parameters(), lr = hp.learning_rate, weight_decay = hp.weight_decay)

        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])

        root = "./data"
        dataset = CIFAR10(root = root, download = True, transform = transform)
        inpaintDataset = InpaintingDataset(dataset)
        train_dataset, val_dataset = torch.utils.data.random_split(inpaintDataset, [len(dataset) * 4 // 5, len(dataset) * 1 // 5])
        train_dataloader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hp.batch_size, shuffle=False)
        
        model = model.to(hp.device)
        self.train(model, hp, train_dataloader, criterion, optimizer)

if __name__ == "__main__":
    obj = runModel()
    obj.run()
