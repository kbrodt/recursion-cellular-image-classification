import torch
import torch.utils.data as D

import rxrx.io as rio

from transforms import train_transform, dev_transform

    
class ImagesSingleExperimentDS(D.Dataset):
    def __init__(self, df, mode, root, transform, with_plates=True):
        self.records = df.to_records(index=False)
        self.root = root
        self.mode = mode
        self.transform = transform
        self.with_plates = with_plates

    def __getitem__(self, index):
        item = self.records[index]
        img = rio.load_site(item.dataset, item.experiment, item.plate, item.well, item.site, base_path=self.root)
        label = item.sirna if self.mode == 'train' else item.id_code
        out = self.transform(img, item.experiment)
        if self.with_plates:
            p = torch.zeros(4, dtype=torch.float32)
            p[item.plate - 1] = 1.0
            out = out, p
        
        return out, label

    def __len__(self):
        return len(self.records)
    
    
def get_loader(dfs, mode, transform, root, ss=(1, 2), batch_size=8, shuffle=False, with_plates=True):
    if len(ss) == 1:
        dfs = [df[df.site == ss[0]].copy() for df in dfs]

    datasets = [ImagesSingleExperimentDS(df=df[df.plate == p], mode=mode, root=root, transform=transform,
                                         with_plates=with_plates)
                for df in dfs for p in range(1, 4 + 1)]
    loaders = [D.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)
               for ds in datasets]
    
    return loaders


def get_loaders(df, train_exp_names, dev_exp_names, root, batch_size=8, n_gpu=1, with_plates=True):
    train_exp = [df[df['experiment'] == exp] for exp in train_exp_names]
    dev_exp = [df[df['experiment'] == exp] for exp in dev_exp_names]

    loaders_train = get_loader(train_exp, mode='train', transform=train_transform,
                               root=root, ss=(1, 2), batch_size=batch_size*n_gpu, shuffle=True, with_plates=with_plates)
    loaders_dev1 = get_loader(dev_exp, mode='train', transform=dev_transform, 
                              root=root, ss=(1,), batch_size=batch_size*n_gpu, shuffle=False, with_plates=with_plates)
    loaders_dev2 = get_loader(dev_exp, mode='train', transform=dev_transform, 
                              root=root, ss=(2,), batch_size=batch_size*n_gpu, shuffle=False, with_plates=with_plates)
    
    return loaders_train, loaders_dev1, loaders_dev2


def get_test_loader(dfs, mode, transform, root, ss=(1, 2), batch_size=8, with_plates=True):
    if len(ss) == 1:
        dfs = [df[df.site == ss[0]].copy() for df in dfs]
        
    datasets = [ImagesSingleExperimentDS(df=df[df.plate == p], mode=mode, root=root, transform=transform,
                                         with_plates=with_plates)
                for df in dfs for p in range(1, 4 + 1)]
    loaders = [D.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
               for ds in datasets]
    
    return loaders


def get_test_loaders(df, exp_names, root, batch_size=8, with_plates=True):
    exps = [df[df['experiment'] == exp] for exp in exp_names]

    loaders_test1 = get_test_loader(exps, mode='test', transform=dev_transform, 
                                    root=root, ss=(1,), batch_size=batch_size, with_plates=with_plates)
    loaders_test2 = get_test_loader(exps, mode='test', transform=dev_transform, 
                                    root=root, ss=(2,), batch_size=batch_size, with_plates=with_plates)
    
    return loaders_test1, loaders_test2
