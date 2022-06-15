import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.model_selection import train_test_split


class Rbp24Dataset(Dataset):

    def __init__(self, df, tokenizer, longest_seq, transform=None):
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform
        self.longest = longest_seq

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        df = self.df

        seq = df['seq'][idx].lower()
        label = df['label'][idx]

        tokenized_seq = self.tokenizer.encode(seq).ids
        padded_seq = np.pad(tokenized_seq, (0, self.longest - len(tokenized_seq)))

        sample = {'seq':padded_seq, 'label':label}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

class ToTensor(object):
    "Convert both seq and label to Tensors"

    def __call__(self, sample):

        seq, label = sample['seq'], sample['label']

        sample = {'seq': torch.tensor(seq, dtype=torch.long),
                  'label': torch.tensor(label)}
        
        return sample


def collate_predict(batch):
    return [item['seq'] for item in batch]


def make_datasets(train_df, test_df, tokenizer, longest_seq, val_size):

    trainset = Rbp24Dataset(train_df, tokenizer, longest_seq, transform=transforms.Compose([ToTensor()]))
    testset = Rbp24Dataset(test_df, tokenizer, longest_seq, transform=transforms.Compose([ToTensor()]))

    train_labels = [int(trainset[i]['label']) for i in range(len(trainset)-1)]
    train_idx, val_idx= train_test_split(np.arange(len(train_labels)), test_size=val_size, shuffle=True, stratify=train_labels)

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    return trainset, testset, train_sampler, val_sampler


def make_dataloaders(train_df, test_df, tokenizer, longest_seq, batch_size, num_workers, val_split):

    trainset, testset, train_sampler, val_sampler = make_datasets(train_df, test_df, tokenizer, longest_seq, val_split)

    trainloader = DataLoader(trainset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers)
    valloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, sampler=val_sampler, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return trainloader, valloader, testloader