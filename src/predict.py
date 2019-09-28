import argparse
from collections import Counter
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

import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch
import torch.nn as nn
import tqdm

sys.path.append('rxrx1-utils')

import rxrx.io as rio
from dataset import get_loaders, get_test_loaders
from model import get_model


def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    arg = parser.add_argument
    arg('--data-path',  default=Path('data'))
    arg('--model-name', type=str, default='densenet121')
    arg('--exp-suffix', type=str, default='bce_wp_fbn_SUB_ACC16')
    
    arg('--batch-size', type=int, default=128)
 
    arg('--seed', type=int, default=314159)
    
    arg('--gpus', type=str)
    arg('--fp16', action='store_true')
    arg('--with-plates', action='store_true')

    args = parser.parse_args()
    
    return args


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def to(xs, device):
    if isinstance(xs, tuple) or isinstance(xs, list):
        return tuple(x.to(device) for x in xs)
    
    return xs.to(device)
    
    
def epoch_step(dataloaders, desc, net, criterion, device, with_preds=False, opt=None, fp16=True):
    is_train = opt is not None
        
    with tqdm.tqdm(it.chain(*dataloaders), desc=desc, mininterval=2, leave=False) as pbar:
        loc_loss = loc_acc = n = 0
        if with_preds:
            loc_logits, loc_targets, loc_plates = [], [], []
        for x, y in pbar:
            if with_preds:
                loc_targets.extend(y.cpu().numpy())
            
            x, y = to(x, device=device), y.to(device)
            logits = net(x)
            if isinstance(x, tuple):
                x, p = x
            loc_plates.extend(p.cpu().numpy())
            
            if with_preds:
                loc_logits.extend(torch.sigmoid(logits).cpu().numpy())
            
            target = torch.zeros_like(logits)
            target[np.arange(x.size(0)), y] = 1
            loss = criterion(logits, target)
            bs = x.size(0)
            loc_loss += loss.item()*bs
            n += bs
            if is_train:
                opt.zero_grad()
                if fp16:
                    with apex.amp.scale_loss(loss, opt) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                opt.step()
                
            y_hat = logits.argmax(dim=-1)
            
            loc_acc += (y == y_hat).sum().item()
                
            pbar.set_postfix(**{'loss': f'{loc_loss/n:.3}', 'acc': f'{loc_acc/n:.3}'})
    if with_preds:
        return loc_loss/n, loc_acc/n, np.array(loc_logits), np.array(loc_targets), np.array(loc_plates).argmax(-1) + 1

    return loc_loss/n, loc_acc/n


def predict(dataloaders, desc, net, device):
    with tqdm.tqdm(it.chain(*dataloaders), desc=desc, mininterval=2, leave=False) as pbar:
        loc_logits, loc_ids, loc_plates = [], [], []
        for x, y in pbar:
            loc_ids.extend(y)
            x = to(x, device=device)
            logits = net(x)
            
            loc_logits.extend(torch.sigmoid(logits).cpu().numpy())
            loc_plates.extend(x[1].cpu().numpy())
            
    return np.array(loc_logits), np.array(loc_ids), np.array(loc_plates).argmax(-1) + 1


def fix_preds(preds):
    preds_sparse = sps.csr_matrix(preds).tocoo()
    
    cls_cnt = Counter()
    ids_cnt = set()
    leak_preds = []
    sorted_preds = np.argsort(preds_sparse.data)[::-1]
    for k in range(3):
        if len(leak_preds) == len(preds):
            break
        for ind in sorted_preds:
            p = preds_sparse.data[ind]
            ids = preds_sparse.row[ind]
            cls = preds_sparse.col[ind]

            if cls_cnt[cls] > k:
                continue
            if ids in ids_cnt:
                continue
            ids_cnt.add(ids)
            cls_cnt[cls] += 1

            leak_preds.append((ids, cls, p))
            if len(leak_preds) == len(preds):
                break

    leaked = pd.DataFrame(leak_preds, columns=['id', 'c', 'p'])\
               .sort_values(['id', 'p'], ascending=[True, False])
                
    return leaked
    
    
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
    data_path = args.data_path
    with_plates = args.with_plates
    model_name = args.model_name
    exp_suffix = args.exp_suffix
    bss = list(range(32, 129))
    cell_types = ['HEPG2', 'HUVEC', 'RPE', 'U2OS']
    root = Path('_'.join([model_name, exp_suffix]))
    cell_to_model = {
        'HEPG2': root / 'HEPG2' / 'seq_train_dev',
        'HUVEC': root / 'HUVEC' / 'seq_train_dev',
        'RPE': root / 'RPE' / 'seq_train_dev',
        'U2OS': root / 'U2OS' / 'seq_train_dev',
    }
    
    dev_tgs, dev_predictions, dev_predictions_fixed = [], [], []
    test_ids, predictions, predictions_fixed = [], [], []
    all_predictions = []
    for CELL_TYPE in cell_types:
        criterion = nn.BCEWithLogitsLoss()
        
        df_ft = rio.combine_metadata(base_path=data_path)
        df_ft.reset_index(inplace=True)
        df_ft = df_ft[(df_ft.well_type == WELL_TYPE) & (df_ft.dataset == 'train')].copy()
        signle_df_ft = df_ft[df_ft['cell_type'] == CELL_TYPE].copy()

        NUM_CLASSES_FT = len(signle_df_ft.sirna.unique())

        signle_df_ft.sirna = signle_df_ft.sirna.apply(np.int64)

        train_exp_names_ft = sorted(signle_df_ft.experiment.unique())
        dev_exp_names_ft = train_exp_names_ft[-1:]
        train_exp_names_ft = train_exp_names_ft[:-1]
        print(train_exp_names_ft)
        print(dev_exp_names_ft)
        
        model_ft2 = get_model(name=model_name, num_classes=NUM_CLASSES_FT, with_plates=with_plates).to(device)
        path_to_pretrained2 = cell_to_model[CELL_TYPE] / 'model.pt'
        state_dict = torch.load(path_to_pretrained2)['state_dict']
        model_ft2.load_state_dict(state_dict, strict=False)
        FP16 = args.fp16 and IS_APEX
        if FP16:
            model_ft2 = apex.amp.initialize(model_ft2, opt_level='O1')
        if n_gpu > 1:
            model_ft2 = nn.parallel.DataParallel(model_ft2)
        
        loc_tgs = []
        loc_dev_preds = []
        loc_dev_preds_fixed = []
        for exp in dev_exp_names_ft:
            print(f'exp: {exp}')
            dev_predictions_bs = []
            for bs in bss:
                print(f'batch: {bs}')
                train_loaders_ft, dev_loaders1_ft, dev_loaders2_ft = get_loaders(signle_df_ft, train_exp_names_ft, [exp],
                                                                                 root=data_path, batch_size=bs*n_gpu,
                                                                                 with_plates=with_plates)
                with torch.no_grad():
                    loss, acc, preds1, targets1, plates1 = epoch_step(dev_loaders1_ft,
                                                                      f'[ Validating {CELL_TYPE} 1 ({exp}/{bs}).. ]',
                                                                      net=model_ft2, criterion=criterion, 
                                                                      device=device,
                                                                      with_preds=True, opt=None, fp16=FP16)
                    print(f'loss site 1: {loss:.4} ({len(preds1)})')
                    print(f'acc site 1: {acc:.4}')

                    loss, acc, preds2, targets2, plates2 = epoch_step(dev_loaders2_ft,
                                                                      f'[ Validating {CELL_TYPE} 2 ({exp}/{bs}).. ]',
                                                                      net=model_ft2, criterion=criterion,
                                                                      device=device,
                                                                      with_preds=True, opt=None, fp16=FP16)
                    print(f'loss site 2: {loss:.4}')
                    print(f'acc site 2: {acc:.4}')

                    assert (targets1 == targets2).all()
                    assert (plates1 == plates2).all()
                    preds = np.mean(np.stack([preds1, preds2]), axis=0)
                    dev_predictions_bs.append(preds)
                    acc = (preds.argmax(-1) == targets1).mean()
                    print(f'acc: {acc:.4}')

                print()
            loc_tgs.extend(targets1)
            preds = np.mean(np.array(dev_predictions_bs), axis=0)
            print(f'mean over batches: {(preds.argmax(-1) == targets1).mean():.4} ({len(preds)})')
            loc_dev_preds.extend(preds.argmax(-1))
            fixed_preds = fix_preds(preds)
            assert len(fixed_preds) == len(preds), f'{len(fixed_preds)}'
            print(f'mean over batches (fixed): {(fixed_preds.c.values == targets1).mean():.4}')
            loc_dev_preds_fixed.extend(fixed_preds.c.values)
            
        dev_tgs.extend(loc_tgs)
        dev_predictions.extend(loc_dev_preds)
        dev_predictions_fixed.extend(loc_dev_preds_fixed)

        test_df = rio.combine_metadata(base_path=data_path)
        test_df.reset_index(inplace=True)
        test_df = test_df[(test_df.well_type == WELL_TYPE) & (test_df.dataset == 'test')].copy()
        to_test = test_df[test_df['cell_type'] == CELL_TYPE].copy()

        loc_ids = []
        loc_preds = []
        loc_preds_fixed = []
        loc_preds_all = []
        for exp in to_test.experiment.unique():
            print(f'exp: {exp}')
            predictions_bs = []
            for bs in bss:
                print(f'batch: {bs}')
                test_loaders1, test_loaders2 = get_test_loaders(to_test, [exp],
                                                                root=data_path, batch_size=bs*n_gpu, with_plates=with_plates)
                with torch.no_grad():
                    preds1, ids1, plates1 = predict(test_loaders1, f'[ Testing {CELL_TYPE} 1 ({exp}/{bs}).. ]',
                                                    net=model_ft2, device=device)
                    print(f'len {len(preds1)}')
                    preds2, ids2, plates2 = predict(test_loaders2, f'[ Testing {CELL_TYPE} 2 ({exp}/{bs}).. ]', 
                                                    net=model_ft2, device=device)

                    assert (ids1 == ids2).all()
                    assert (plates1 == plates2).all()
                    preds = np.mean(np.stack([preds1, preds2]), axis=0)
                    assert len(ids1) == len(preds)
                    predictions_bs.append(preds)

            loc_ids.extend(ids1)
            
            preds = np.mean(np.array(predictions_bs), axis=0)
            loc_preds.extend(preds.argmax(-1))
            fixed_preds = fix_preds(preds)
            assert len(fixed_preds) == len(preds)
            loc_preds_fixed.extend(fixed_preds.c.values)
            
            loc_preds_all.extend(preds)
            
        test_ids.extend(loc_ids)
        predictions.extend(loc_preds)
        predictions_fixed.extend(loc_preds_fixed)
        all_predictions.extend(loc_preds_all)
    
        assert len(test_ids) == len(predictions) == len(predictions_fixed)
    
    dev_tgs, dev_predictions, dev_predictions_fixed = map(np.array, [dev_tgs, dev_predictions, dev_predictions_fixed])
    all_predictions = np.array(all_predictions)
    print(f'acc           : {(dev_tgs == dev_predictions).mean():.4}')
    print(f'acc (fixed)   : {(dev_tgs == dev_predictions_fixed).mean():.4}')
    to_sub = pd.DataFrame(zip(test_ids, predictions, predictions_fixed, *all_predictions.T,),
                          columns=['id_code', 'sirna', 'sirna_fixed',
                                  ] + [f'p_{i}' for i in range(NUM_CLASSES_FT)])
    to_sub.to_csv(f'submission_SUB_ACC16_p.csv', index=False)
    
    # plate "leak"
    train_csv = pd.read_csv(data_path / 'train.csv')
    test_csv = pd.read_csv(data_path / 'test.csv')
    test_csv = pd.merge(test_csv, to_sub, how='left', on='id_code')
    sub = pd.read_csv(f'submission_SUB_ACC16_p.csv')
    assert (test_csv.id_code.values == sub.id_code.values).all()
    plate_groups = np.zeros((NUM_CLASSES_FT, 4), int)
    for sirna in range(NUM_CLASSES_FT):
        grp = train_csv.loc[train_csv.sirna == sirna, :].plate.value_counts().index.values
        assert len(grp) == 3
        plate_groups[sirna, 0:3] = grp
        plate_groups[sirna, 3] = 10 - grp.sum()
        
    all_test_exp = test_csv.experiment.unique()

    group_plate_probs = np.zeros((len(all_test_exp), 4))
    for idx in range(len(all_test_exp)):
        preds = sub.loc[test_csv.experiment == all_test_exp[idx], 'sirna_fixed'].values
        pp_mult = np.zeros((len(preds), NUM_CLASSES_FT))
        pp_mult[range(len(preds)), preds] = 1

        sub_test = test_csv.loc[test_csv.experiment == all_test_exp[idx], :]
        assert len(pp_mult) == len(sub_test)

        for j in range(4):
            mask = np.repeat(plate_groups[np.newaxis, :, j], len(pp_mult), axis=0) == \
                   np.repeat(sub_test.plate.values[:, np.newaxis], NUM_CLASSES_FT, axis=1)

            group_plate_probs[idx,j] = np.array(pp_mult)[mask].sum()/len(pp_mult)
    exp_to_group = group_plate_probs.argmax(1)
    
    def select_plate_group(pp_mult, idx):
        sub_test = test_csv.loc[test_csv.experiment == all_test_exp[idx],:]
        assert len(pp_mult) == len(sub_test)
        mask = np.repeat(plate_groups[np.newaxis, :, exp_to_group[idx]], len(pp_mult), axis=0) != \
               np.repeat(sub_test.plate.values[:, np.newaxis], NUM_CLASSES_FT, axis=1)
        pp_mult[mask] = 0

        return pp_mult
    
    for idx in range(len(all_test_exp)):
        indices = (test_csv.experiment == all_test_exp[idx])

        preds = test_csv[indices].copy()
        preds = preds[[f'p_{i}' for i in range(NUM_CLASSES_FT)]].values

        preds = select_plate_group(preds, idx)
        sub.loc[indices, 'sirna_leak'] = preds.argmax(1)
        
        preds_fixed = fix_preds(preds)
        assert len(preds_fixed) == len(preds)
        sub.loc[indices, 'sirna_leak_fixed'] = preds_fixed.c.values
        
    sub.to_csv(f'submission_SUB_ACC16_p_leak.csv', index=False)

    
if __name__ == '__main__':
    main()
