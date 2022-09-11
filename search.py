import argparse
import os
import datetime

from tqdm import tqdm
import random
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import json
import model
from loader.data_loader import get_dataloader
import thop
from utils import AverageMeter
BATCHSIZE = 4
N_TRAIN_EXAMPLES = BATCHSIZE * 60
N_VALID_EXAMPLES = BATCHSIZE * 10
DEVICE = 'cuda'
NUM_EPOCHS = 10


def args_pars():
    args_parsar = argparse.ArgumentParser(description="The argument parser for searching most optimal model.")
    args_parsar.add_argument('--trail')

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


def define_model(trial: "optuna.Trial"):
    n_layers = trial.suggest_int("n_layers", 1, 5)
    layers = [model.Conv7X7BnReLU(in_features=3, out_features=64, stride=1)]
    in_features = 64
    # loop
    for i in range(n_layers):
        out_features = trial.suggest_int("n_feature_l{}".format(i), 32, 128, log=True)
        # feature space convolution block
        layer_type = trial.suggest_categorical(f"feature_block_{i}", list(model.FeatureBlocks.keys()))
        layers.append(get_feature_block(layer_type, in_features, out_features))
        if layer_type != 'seblock':
            in_features = out_features

        # reduction space convolution block
        to_reduce = trial.suggest_categorical(f"reduction_{i}", [0, 1])
        if bool(to_reduce):
            out_features = trial.suggest_int("n_reduction_l{}".format(i), 32, 512, log=True)
            layer_type_reduction = trial.suggest_categorical(f"reduction_block_{i}", list(model.ReductionBlock.keys()))
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


def objective(trial: "optuna.Trial"):
    # create model
    model = define_model(trial)

    datasets = get_dataloader(only_dataset=True,
        path='/home/sarvesh/Documents/repositories/sarvesh-personal/grain-classification/Rice_Image_Dataset/data_splits')
    # dataloader
    train_dataset = datasets['train']
    val_dataset = datasets['validation']

    train_loader = DataLoader(
        torch.utils.data.Subset(train_dataset, random.choices(list(range(len(train_dataset))), k=N_TRAIN_EXAMPLES)),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        torch.utils.data.Subset(val_dataset, random.choices(list(range(len(val_dataset))), k=N_VALID_EXAMPLES)),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    # Generate optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    )

    loss = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        los = train_model(model, optimizer, loss, train_loader, epoch)
    if los > 0:
        flops, accuracy = eval_model(model, val_loader)
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


x = datetime.datetime.now()
path = f'./result/v_{x.strftime("%d-%b-%Y_T_%H:%M:%S%p")}'

study = optuna.create_study(directions=["minimize", "maximize"])
study.optimize(objective, n_trials=20, timeout=6000)
print("Number of finished trials: ", len(study.trials))
save_all_trails(study.get_trials(), path)

fig = optuna.visualization.plot_pareto_front(study, target_names=["FLOPS", "accuracy"])
fig.show("browser")
fig.write_html(f"{path}/pareto_visulalization.html")
print(f"Number of trials on the Pareto front: {len(study.best_trials)}")



trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[1])
print(f"Trial with highest accuracy: ")
print(f"\tnumber: {trial_with_highest_accuracy.number}")
print(f"\tparams: {trial_with_highest_accuracy.params}")
print(f"\tvalues: {trial_with_highest_accuracy.values}")

fga = optuna.visualization.plot_param_importances(
    study, target=lambda t: t.values[0], target_name="flops"
)
fga.show("browser")
fga.write_html(f"{path}/plot_param_importance_flops.html")

fg = optuna.visualization.plot_param_importances(
    study, target=lambda t: t.values[1], target_name="accuracy"
)
fg.show("browser")
fg.write_html(f"{path}/plot_param_importance_accuracy.html")
