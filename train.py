import argparse
import os
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch
from torch import nn
from sklearn.metrics import classification_report
import json
import model
from loader.data_loader import get_dataloader
import thop
from utils import AverageMeter

BATCHSIZE = 16
DEVICE = 'cuda'
NUM_EPOCHS = 10
PATH = '/home/sarvesh/Documents/repositories/sarvesh-personal/grain-classification/Rice_Image_Dataset/new_split'


def args_pars():
    args_parsar = argparse.ArgumentParser(description="The argument parser for searching most optimal model.")
    args_parsar.add_argument('--trail',
                             help="specify the json file created using search.py in results folder select the "
                                  "suitable model")
    args_parsar.add_argument('--state-dict',
                             help="path to the state_dict created by the model")
    cfg = args_parsar.parse_args()
    return cfg


def get_feature_block(name, in_features, out_features):
    if name == 'resnetblock':
        return model.FeatureBlocks[name](in_features, out_features)
    elif name == 'seblock':
        return model.FeatureBlocks[name](in_features)
    elif name == 'conv':
        return model.FeatureBlocks[name](in_features, out_features, stride=1)
    elif name == 'shuffleunit':
        return model.FeatureBlocks[name](in_features, out_features, 1)
    else:
        raise Exception("invalid input")


def get_reduction_block(name, in_features, out_features):
    if name == 'resnetblock':
        return model.ReductionBlock[name](in_features, out_features)
    elif name == 'conv':
        return model.ReductionBlock[name](in_features, out_features, stride=2)
    elif name == 'shuffle_unit':
        return model.ReductionBlock[name](in_features, out_features)
    else:
        raise Exception("invalid input")


def define_model(parameters):
    n_layers = parameters["n_layers"]
    layers = [model.Conv7X7BnReLU(in_features=3, out_features=64, stride=1)]
    in_features = 64
    # loop
    for i in range(n_layers):
        out_features = parameters["n_feature_l{}".format(i)]
        # feature space convolution block
        layer_type = parameters[f"feature_block_{i}"]
        layers.append(get_feature_block(layer_type, in_features, out_features))
        if layer_type != 'seblock':
            in_features = out_features

        # reduction space convolution block
        to_reduce = parameters[f"reduction_{i}"]
        if bool(to_reduce):
            out_features = parameters["n_reduction_l{}".format(i)]
            layer_type_reduction = parameters[f"reduction_block_{i}"]
            layers.append(get_reduction_block(layer_type_reduction, in_features, out_features))
            in_features = out_features

    layers.append(nn.AdaptiveAvgPool2d(output_size=1))
    layers.append(model.Flatten())
    layers.append(nn.Linear(in_features, 128))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(128, 5))
    return nn.Sequential(*layers)


def save_trials(result: "optuna.trial._frozen.FrozenTrial", idx, path, attr=None):
    if attr is None:
        attr =['number', 'values', 'params']
    result_dict = dict()
    for i in attr:
        try:
            if not callable(getattr(result, i)) and not (i.startswith('_')):
                result_dict[i] = getattr(result, i)
        except RuntimeError:
            continue

    with open(f"{path}/result_{idx}.json", 'w') as f:
        f.write(json.dumps(result_dict))


def save_all_trails(results: list, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for n, result in enumerate(results):
        save_trials(result, n, path)


def main(parameters, state_dict):
    # create model
    init_params = parameters['params']
    model = define_model(init_params)
    if state_dict is not None:
        model.load_state_dict(torch.load(state_dict))
    data_sets, loaders = get_dataloader(path=PATH, batch_size=BATCHSIZE, only_dataset=False, num_workers=8)
    # dataloader
    train_loader = loaders['train']
    val_loader = loaders['validation']

    # Generate optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )

    lr_schedular = ReduceLROnPlateau(optimizer, factor=0.1, patience=2,verbose=True)
    loss = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        los = train_model(model, optimizer, loss, train_loader, epoch)
        lr_schedular.step(los, epoch)
        accuracy = eval_model(model, val_loader)
        print(accuracy)
        torch.save(model.state_dict(), f'state_dict/model_{epoch}.pth')
    if los > 0:
        flops, accuracy = eval_model(model, loaders['test'])
    else:
        flops, accuracy = -1, 0

    return flops, accuracy


def train_model(model, optimizer, loss, train_loader, epoch):
    model.train()
    model.to(DEVICE)
    avg_loss = AverageMeter()
    loop = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(loop):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        try:
            lo = loss(model(data), target)
        except RuntimeError as e:
            return -1
        lo.backward()
        optimizer.step()
        avg_loss.update(float(lo))
        del data
        torch.cuda.empty_cache()
        loop.set_description(f'Epoch[{epoch}/{NUM_EPOCHS}](train)')
        loop.set_postfix(avg_loss=avg_loss.avg)
    return avg_loss.avg


def eval_model(model, valid_loader):
    model.eval()
    predictions = []
    labels = []
    loop = tqdm(valid_loader)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loop):
            data, target = data.to(DEVICE), target.to(DEVICE)
            pred = model(data).argmax(dim=1, keepdim=True)
            predictions.extend(torch.argmax(pred, 1).cpu().detach().tolist())
            labels.extend(target.cpu().detach().tolist())
            loop.set_description('validation')
    report_dict = classification_report(labels, predictions, output_dict=True)

    flops, _ = thop.profile(model, inputs=(torch.randn(16, 3, 224, 224).to(DEVICE),), verbose=False)
    return flops, report_dict['accuracy']


if __name__ == '__main__':
    cfg = args_pars()
    with open(cfg.trail,'r') as f:
        trail_info = json.load(f)
    main(trail_info, cfg.state_dict)
