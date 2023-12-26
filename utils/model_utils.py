import os

import pandas as pd
import torch
import torch.optim
import torch.utils.data
import torchvision.models as models


def ft_train(args, train_loader, model, criterion, optimizer, scaler, epoch):
    model.train()

    all_targets = []
    all_preds = []
    running_loss = 0.0
    total_step = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        total_step += 1
        # learning_rate_decay(optimizer, len(train_loader) * epoch + i, args.lr)

        inputs = inputs.to(args.default_device)
        labels = labels.to(args.default_device)

        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        _, preds = torch.max(outputs.data, 1)

        all_targets += list(labels.cpu().numpy())
        all_preds += list(preds.cpu().numpy())

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'.format(epoch, batch_idx, len(train_loader)))
        if args.debug:
            break

    return pd.DataFrame({'preds': all_preds, 'labels': all_targets}), running_loss / total_step

@torch.no_grad()
def validation(args, test_loader, model, criterion):
    all_targets = []
    all_preds = []

    model.eval()
    running_loss = 0.0
    total_step = 0
    for batch_idx, (inputs, labels) in enumerate(test_loader):
        total_step += 1
        inputs = inputs.to(args.default_device)
        labels = labels.to(args.default_device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs.data, 1)

        all_targets += list(labels.cpu().numpy())
        all_preds += list(preds.cpu().numpy())

        running_loss += loss.item()

    return pd.DataFrame({'preds': all_preds, 'labels': all_targets}), running_loss / total_step

@torch.no_grad()
def test(args, test_loader, model):
    all_targets = []
    all_preds = []

    model.eval()
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(args.default_device)
        labels = labels.to(args.default_device)

        outputs = model(inputs)

        _, pred = torch.max(outputs.data, 1)

        all_targets += list(labels.cpu().numpy())
        all_preds += list(pred.cpu().numpy())

    return pd.DataFrame({'preds': all_preds, 'labels': all_targets})


def load_ft_model(args):
    path = args.model_path
    if args.model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif args.model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif args.model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    else:
        raise NotImplementedError

    num_fc = model.fc.in_features
    if args.dataset == 'NCT':
        model.fc = torch.nn.Linear(num_fc, 9)
    else:
        model.fc = torch.nn.Linear(num_fc, 2)

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location='cpu')  # resnet.layer4.2.bn2.num_batches_tracked
        model_dict = model.state_dict()  # layer4.2.bn2.num_batches_tracked

        def rename_key(key):
            if not 'module.resnet.' in key:
                return key
            return ''.join(key.split('module.resnet.'))

        checkpoint_dict = {}
        for key, val in checkpoint['net'].items():
            # for key, val in checkpoint['state_dict'].items():
            checkpoint_dict[rename_key(key)] = val
        model_dict.update(checkpoint_dict)
        missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)
        print(f'missing keys:{missing_keys}, unexpected_keys:{unexpected_keys}')
        # model.load_state_dict(model_dict)
        print("loaded")
    else:
        # model = None
        print("=> no checkpoint found at '{}".format(path))
    return model
