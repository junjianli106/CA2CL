import argparse
import time

import numpy as np
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.multiprocessing
import torchvision.transforms as transforms
from torch.nn.parallel import DataParallel

torch.multiprocessing.set_sharing_strategy('file_system')

from utils.utils import *
from utils.datasets import *
from utils.yaml_config_hook import *
from utils.save_model import *
from sync_batchnorm import convert_model
from modules import resnet, network, contrastive_loss
from modules.Transform import Transform

from utils.common import *
from utils.model_utils import *

from modules.transforms import build_transform#,ImageFolder

def train(args, scaler):
    loss_epoch = 0
    for step, (x, _) in enumerate(data_loader):
        x0 = x[0].to(args.default_device)
        x1 = x[1].to(args.default_device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(True):
            z_i, z_i_groups = model(x0)
            z_j, z_j_groups = model(x1)
            loss = criterion_instance([z_i, z_j])
            if args.do_ca:
                ca_loss = CAM(z_i_groups, z_j_groups, args.ca_t, args)
                loss_ins_ca = loss + args.Lambda * (ca_loss)

        if args.do_ca:
            scaler.scale(loss_ins_ca).backward(retain_graph=True)
        else:
            if args.do_adv:
                scaler.scale(loss).backward(retain_graph=True)
            else:
                scaler.scale(loss).backward()

        if args.do_adv:
            advm.attack()
            with torch.cuda.amp.autocast(True):
                z_i_adv, z_i_groups_adv = model(x0)
                loss_instance_adv = criterion_instance([z_j, z_i_adv])

            advm.restore()
            scaler.scale(loss_instance_adv).backward()

        scaler.step(optimizer)
        scaler.update()

        model.zero_grad()

        if args.do_adv:
            if step % 10 == 0:
                if args.do_ca:
                    print(
                        f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss.item()}",
                        f"Step [{step}/{len(data_loader)}]\t loss_adv: {loss_instance_adv.item()}",
                        f"Step [{step}/{len(data_loader)}]\t ca_loss: {loss_ins_ca.item()}")
                else:
                    print(
                        f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss.item()}",
                        f"Step [{step}/{len(data_loader)}]\t loss_adv: {loss_instance_adv.item()}")
        else:
            if step % 50 == 0:
                print(
                    f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss.item()}")

        if args.do_adv:
            if args.do_ca:
                loss_epoch += loss_ins_ca.item() + loss_instance_adv.item()
            else:
                loss_epoch += loss.item() + loss_instance_adv.item()
        else:
            loss_epoch = loss.item()
    return loss_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_redis', action='store_true')
    parser.add_argument('--default_device', type=str, default='cuda', help='default_device')
    parser.add_argument('--arch', type=str, default='resnet18', help='model architecture')
    parser.add_argument('--comet_name', type=str, default='pretrain',
                        help='对本次实验的命名, 用于comet.ml')
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    args.model_path = f"save/{args.dataset}/pretrain/{args.resnet}_bs_{args.batch_size}_img_size_{args.image_size}_lr_{args.learning_rate}_\
                        wd_{args.weight_decay}_temperature_{args.instance_temperature}_adv_eps_{args.epsilon}_ca_{args.Lambda}_strong_weak"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    transform = Transform(init_size=224)

    dataset = CRC_Dataset(
        dataset_path=args.dataset_path,
        transform=transform
    )

    print('dataset:', len(dataset))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True
    )

    # initialize model
    res = resnet.get_resnet(args.resnet)
    model = network.Network(res, args.feature_dim)

    load_imageweights = True
    if load_imageweights:
        if args.resnet == 'ResNet18' and not os.path.isfile('./save/resnet18-f37072fd.pth'):
            print('Downloading imagenet weights...')
            os.system('wget https://download.pytorch.org/models/resnet18-f37072fd.pth')
        if args.resnet == 'ResNet34' and not os.path.isfile('./save/resnet34-333f7ec4.pth'):
            print('Downloading imagenet weights...')
            os.system('wget https://download.pytorch.org/models/resnet34-333f7ec4.pth')
        if args.resnet == 'ResNet50' and not os.path.isfile('./save/resnet50-19c8e357.pth'):
            print('Downloading imagenet weights...')
            os.system('wget https://download.pytorch.org/models/resnet50-19c8e357.pth')

        if args.resnet ==  'ResNet18':
            print('Loading imagenet ResNet18 weights...')
            checkpoint = torch.load('./save/resnet18-f37072fd.pth', map_location='cpu')
        elif args.resnet == 'ResNet34':
            print('Loading imagenet ResNet34 weights...')
            checkpoint = torch.load('./save/resnet34-333f7ec4.pth', map_location='cpu')
        elif args.resnet == 'ResNet50':
            print('Loading imagenet ResNet50 weights...')
            checkpoint = torch.load('./save/resnet50-19c8e357.pth', map_location='cpu')
        else:
            raise NotImplementedError
        model_dict = model.state_dict()

        def rename_key(key):
            if not 'resnet' in key:
                return 'resnet.' + key
            return key

        checkpoint_dict = {}
        for key, val in checkpoint.items():
            checkpoint_dict[rename_key(key)] = val
        model_dict.update(checkpoint_dict)
        missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)
        print(f'missing keys:{missing_keys}, unexpected_keys:{unexpected_keys}')

    dataparallel = 1
    if dataparallel:
        model = convert_model(model)
        model = DataParallel(model)
    model = model.to(args.default_device)
    print('load the model successfully')

    # optimizer / loss
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # loss_device = torch.device("cuda")
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, args.default_device).to(
        args.default_device)
    print('load the criterion_instance successfully')

    if args.do_adv:
        advm = AdvModule(model, dataparallel, epsilon=args.epsilon, emb_name='resnet.conv1.weight')
    print('start training')
    for epoch in range(args.start_epoch, args.epochs):
        start_ts = time.time()
        lr = optimizer.param_groups[0]["lr"]
        adjust_learning_rate(optimizer, epoch, args)
        loss_epoch = train(args, scaler)
        if epoch % 10 == 0:
            save_model(args, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}, cost:{time.time() - start_ts}")
    save_model(args, model, optimizer, args.epochs)
