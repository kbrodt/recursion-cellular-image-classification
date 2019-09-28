import argparse
import itertools as it
import os
from pathlib import Path
import random
import sys

try:
    import apex
    IS_APEX = True
except:
    IS_APEX = False

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm

sys.path.append('rxrx1-utils')

import rxrx.io as rio
from dataset import get_loaders
from model import get_model


def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    arg = parser.add_argument
    arg('--data_path',  default=Path('data'))
    
    arg('--model-name', type=str, default='densenet121')
    arg('--exp-suffix', type=str, default='bce_wp_fbn_SUB_ACC16')

    arg('--cell-type',  type=str, choices=['HEPG2', 'HUVEC', 'RPE', 'U2OS'])
    arg('--with-plates', action='store_true', default=True)
    arg('--batch-size', type=int, default=8)
    arg('--lr',         type=float, default=1e-3)
    arg('--epochs',     type=int, default=100)
    arg('--gamma',      type=float, default=0.1)
 
    arg('--seed', type=int, default=314159)
    
    arg('--gpus', type=str)
    arg('--fp16', action='store_true')

    args = parser.parse_args()
    
    return args


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_history(history, path):
    ep = np.argmax(history['acc']['dev'])
    acc = history['acc']['dev'][ep]
    fig = plt.figure(figsize=(12, 4))
    for i, (key, value) in enumerate(history.items()):
        plt.subplot(1, 2, i + 1)
        if i == 1:
            plt.title(f'{acc:.3} on {ep}')
        for k, v in value.items():
            plt.plot(v, label=f'{k} {v[-1]:.3}')
        plt.xlabel('#epoch')
        plt.ylabel(key)
        plt.legend()
        plt.grid(ls='--')
    
    plt.savefig(path / 'evolution.png')
    plt.close(fig)
    
    
def to(xs, device):
    if isinstance(xs, tuple) or isinstance(xs, list):
        return tuple(x.to(device) for x in xs)
    
    return xs.to(device)

    
def epoch_step(dataloaders, desc, net, criterion, device, with_preds=False, opt=None, fp16=True, n_accum=16):
    is_train = opt is not None
        
    with tqdm.tqdm(it.chain(*dataloaders), desc=desc, mininterval=2, leave=False) as pbar:
        loc_loss = loc_acc = n = 0
        accum = 1
        if with_preds:
            loc_logits, loc_targets = [], []
        for x, y in pbar:
            if with_preds:
                loc_targets.extend(y.numpy())
            
            x, y = to(x, device=device), y.to(device)
            logits = net(x)
            if isinstance(x, tuple):
                x, _ = x
            
            if with_preds:
                loc_logits.extend(torch.sigmoid(logits).cpu().numpy())
            
            target = torch.zeros_like(logits)
            target[np.arange(x.size(0)), y] = 1
            loss = criterion(logits, target)
            bs = x.size(0)
            loc_loss += loss.item()*bs
            n += bs
            if is_train:
                if fp16:
                    with apex.amp.scale_loss(loss, opt) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                if accum % n_accum == 0:
                    opt.step()
                    opt.zero_grad()
                    accum = 1
                else:
                    accum += 1
                
            y_hat = logits.argmax(dim=-1)
            loc_acc += (y == y_hat).sum().item()
                
            pbar.set_postfix(**{'loss': f'{loc_loss/n:.3}', 'acc': f'{loc_acc/n:.3}'})
        if is_train and accum > 1:
            opt.step()
            opt.zero_grad()
    if with_preds:
        return loc_loss/n, loc_acc/n, np.array(loc_logits), np.array(loc_targets)

    return loc_loss/n, loc_acc/n


def train_model(loaders, model, criterion, opt, path, device, fp16=True, epochs=100, history=None, scheduler=None):
    if history is None:
        history = {
            'loss': {'train': [], 'dev1': [], 'dev2': []},
            'acc': {'train': [], 'dev1': [], 'dev2': [], 'dev': []}
        }
        max_acc = 0
        start_epoch = 0
    else:
        max_acc = max(history['acc']['dev'])
        start_epoch = len(history['acc']['dev'])
        
    train_loaders, dev_loaders1, dev_loaders2 = loaders
    for epoch in range(start_epoch + 1, epochs + 1):
        loss, acc = epoch_step(train_loaders, f'[ Training {epoch}/{epochs}.. ]',
                               net=model, criterion=criterion, device=device, with_preds=False, opt=opt, fp16=fp16)
        history['loss']['train'].append(loss)
        history['acc']['train'].append(acc)
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            loss, acc, preds1, targets1 = epoch_step(dev_loaders1, f'[ Validating {epoch}/{epochs}.. ]',
                                                     net=model, criterion=criterion,
                                                     device=device, with_preds=True, opt=None, fp16=fp16)
            history['loss']['dev1'].append(loss)
            history['acc']['dev1'].append(acc)

            loss, acc, preds2, targets2 = epoch_step(dev_loaders2, f'[ Validating {epoch}/{epochs}.. ]',
                                                     net=model, criterion=criterion,
                                                     device=device, with_preds=True, opt=None, fp16=fp16)
            history['loss']['dev2'].append(loss)
            history['acc']['dev2'].append(acc)

            assert (targets1 == targets2).all()
            preds = np.mean(np.stack([preds1, preds2]), axis=0)
            acc = (preds.argmax(-1) == targets1).mean()
            history['acc']['dev'].append(acc)

        if history['acc']['dev'][-1] > max_acc:
            max_acc = history['acc']['dev'][-1]
            torch.save({
                'state_dict': model.state_dict(),
                'history': history,
                'epoch': epoch,
            }, path / 'model.pt')
        torch.save({
            'state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict(),
            'history': history,
            'epoch': epoch,
        }, path / 'last.pt')

        plot_history(history, path)
        
    return history, max_acc
    
    
def main():
    args = get_args()
    print(args)
    
    gpus = args.gpus
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    device = 'cuda'
    
    n_gpu = len(gpus.split(','))
    set_seeds(args.seed)
    WELL_TYPE = 'treatment'
    CELL_TYPE = args.cell_type
    data_path = args.data_path
    with_plates = args.with_plates
    model_name = args.model_name
    exp_suffix = args.exp_suffix
    FP16 = args.fp16 and IS_APEX
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    criterion = nn.BCEWithLogitsLoss()

    df = rio.combine_metadata(base_path=data_path)
    df.reset_index(inplace=True)
    df = df[(df.well_type != WELL_TYPE)].copy()
    signle_df = df[df['cell_type'] == CELL_TYPE].copy()
    NUM_CLASSES = len(signle_df.sirna.unique())

    mapping = {cl: ind for ind, cl in enumerate(sorted(signle_df.sirna.unique()))}
    signle_df.sirna = signle_df.sirna.apply(lambda x: mapping[x])

    train_exp_names = sorted(signle_df[signle_df.dataset == 'train'].experiment.unique())
    dev_exp_names = sorted(signle_df[signle_df.dataset == 'test'].experiment.unique())
    
    train_loaders, dev_loaders1, dev_loaders2 = get_loaders(signle_df, train_exp_names, dev_exp_names,
                                                            root=data_path, batch_size=batch_size, n_gpu=n_gpu,
                                                            with_plates=with_plates)
    
    
    path_to_exp = Path('_'.join([model_name, exp_suffix])) / CELL_TYPE / 'seq_pretrain'
    if not path_to_exp.exists():
        path_to_exp.mkdir(parents=True)
        
    model = get_model(name=model_name, num_classes=NUM_CLASSES, with_plates=with_plates).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    if FP16:
        model, opt = apex.amp.initialize(model, opt, opt_level='O1')
    if n_gpu > 1:
        model = nn.parallel.DataParallel(model)
    
    scheduler = None
    train_model((train_loaders, dev_loaders1, dev_loaders2),
                model=model, criterion=criterion, opt=opt,
                path=path_to_exp, device=device,
                fp16=FP16, epochs=epochs, scheduler=scheduler)
    
#     # pretrain head
    path_to_pretrained = path_to_exp / 'model.pt'
    df_ft = rio.combine_metadata(base_path=data_path)
    df_ft.reset_index(inplace=True)
    df_ft = df_ft[(df_ft.well_type == WELL_TYPE) & (df_ft.dataset == 'train')].copy()
    signle_df_ft = df_ft[df_ft['cell_type'] == CELL_TYPE].copy()

    NUM_CLASSES_FT = len(signle_df_ft.sirna.unique())

    signle_df_ft.sirna = signle_df_ft.sirna.apply(np.int64)

    train_exp_names_ft = sorted(signle_df_ft.experiment.unique())
    dev_exp_names_ft = train_exp_names_ft[-1:]
    train_exp_names_ft = train_exp_names_ft[:-1]

    train_loaders_ft, dev_loaders1_ft, dev_loaders2_ft = get_loaders(signle_df_ft, train_exp_names_ft, dev_exp_names_ft,
                                                                     root=data_path, batch_size=batch_size, n_gpu=n_gpu,
                                                                     with_plates=with_plates)
    path_to_exp_ft = Path('_'.join([model_name, exp_suffix])) / CELL_TYPE / 'seq_train_head'
    if not path_to_exp_ft.exists():
        path_to_exp_ft.mkdir(parents=True)
    model_ft = get_model(name=model_name, num_classes=NUM_CLASSES_FT, with_plates=with_plates).to(device)
    state_dict = torch.load(path_to_pretrained)['state_dict']
    state_dict.pop('classifier.weight')
    state_dict.pop('classifier.bias')
    model_ft.load_state_dict(state_dict, strict=False)
    for n, p in model_ft.named_parameters():
        if not n.startswith('classifier'):
            p.requires_grad = False
    opt_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=lr, amsgrad=True)

    if FP16:
        model_ft, opt_ft = apex.amp.initialize(model_ft, opt_ft, opt_level='O1')
    if n_gpu > 1:
        model_ft = nn.parallel.DataParallel(model_ft)
    scheduler = None
    train_model((train_loaders_ft, dev_loaders1_ft, dev_loaders2_ft),
                model=model_ft, criterion=criterion, opt=opt_ft,
                path=path_to_exp_ft, device=device,
                fp16=FP16, epochs=epochs + 50, scheduler=scheduler)
    
    # finetune whole model
    path_to_exp_ft2 = Path('_'.join([model_name, exp_suffix])) / CELL_TYPE / 'seq_train'
    if not path_to_exp_ft2.exists():
        path_to_exp_ft2.mkdir(parents=True)
    model_ft2 = get_model(name=model_name, num_classes=NUM_CLASSES_FT, with_plates=with_plates).to(device)
    path_to_pretrained2 = path_to_exp_ft / 'model.pt'
    state_dict = torch.load(path_to_pretrained2)['state_dict']
    model_ft2.load_state_dict(state_dict)
    opt_ft2 = torch.optim.Adam(model_ft2.parameters(), lr=lr, amsgrad=True)
    if FP16:
        model_ft2, opt_ft2 = apex.amp.initialize(model_ft2, opt_ft2, opt_level='O1')
    if n_gpu > 1:
        model_ft2 = nn.parallel.DataParallel(model_ft2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt_ft2, milestones=[120, 150], gamma=args.gamma)
    train_model((train_loaders_ft, dev_loaders1_ft, dev_loaders2_ft),
                model=model_ft2, criterion=criterion, opt=opt_ft2,
                path=path_to_exp_ft2, device=device,
                fp16=FP16, epochs=epochs + 75, scheduler=scheduler)

    # finetune on validation
    path_to_exp_ft3 = Path('_'.join([model_name, exp_suffix])) / CELL_TYPE / 'seq_train_dev'
    if not path_to_exp_ft3.exists():
        path_to_exp_ft3.mkdir(parents=True)
    opt_ft2 = torch.optim.Adam(model_ft2.parameters(), lr=1e-5, amsgrad=True)
    if FP16:
        model_ft2, opt_ft2 = apex.amp.initialize(model_ft2, opt_ft2, opt_level='O1')
    if n_gpu > 1:
        model_ft2 = nn.parallel.DataParallel(model_ft2)
    train_model((list(it.chain(train_loaders_ft, dev_loaders1_ft, dev_loaders2_ft)), dev_loaders1_ft, dev_loaders2_ft),
                model=model_ft2, criterion=criterion, opt=opt_ft2,
                path=path_to_exp_ft3, device=device,
                fp16=FP16, epochs=15, scheduler=None)


if __name__ == '__main__':
    main()
