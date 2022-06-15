import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.model_selection import train_test_split


class Rbp24Dataset(Dataset):

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        df = self.df
        seq = df['seq'][idx].upper()
        label = df['label'][idx]

        sample = {'seq':seq, 'label':label}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

class ToOHE(object):
    "Convert seq to One Hot Encoding, convert both seq and label to Tensors"

    def __call__(self, sample):

        seq, label = sample['seq'], sample['label']

        nucleotid = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1], '':[0,0,0,0], 'N':[0.25,0.25,0.25,0.25]}
        seq = np.array([nucleotid[x] for x in seq])

        sample = {'seq': torch.from_numpy(seq).float().permute(1,0),
                  'label': torch.tensor(label)}
        
        return sample


def make_datasets(train_df, test_df, val_size):

    trainset = Rbp24Dataset(train_df, transform=transforms.Compose([ToOHE()]))
    testset = Rbp24Dataset(test_df, transform=transforms.Compose([ToOHE()]))

    train_labels = [int(trainset[i]['label']) for i in range(len(trainset)-1)]
    train_idx, val_idx= train_test_split(np.arange(len(train_labels)), test_size=val_size, shuffle=True, stratify=train_labels)

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    return trainset, testset, train_sampler, val_sampler


def make_dataloaders(train_df, test_df, batch_size, num_workers, val_split):

    trainset, testset, train_sampler, val_sampler = make_datasets(train_df, test_df, val_split)

    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
    valloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, sampler=val_sampler, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return trainloader, valloader, testloader