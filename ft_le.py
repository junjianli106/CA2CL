# Function: fine-tuning and linear evaluation
import warnings

warnings.filterwarnings('ignore')
import argparse

import dataset_loader
from utils.common import *
from utils.model_utils import *

log = get_logger(__name__)


def finetuning(args):
    train_loader, val_loader, test_loader = dataset_loader.get_dataloader(args, mode='fine-tuning')

    model = load_ft_model(args)
    # log.info('args.default_device', args.default_device)

    model = model.to(args.default_device)

    criterion = torch.nn.CrossEntropyLoss().to(args.default_device)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, list(model.parameters())),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = get_scheduler(optimizer, 4e-8)

    # val
    best_val_acc = 0

    early_st = args.early_stop
    scaler = None

    if args.fp16:
        log.info('Use fp16 and loaded')
        scaler = torch.cuda.amp.GradScaler()

    if scaler:
        print(f'scaler: {scaler}')

    for epoch in range(args.ft_epoch):
        log.info('=' * 10 + f'Epoch:{epoch}' + '=' * 10)

        # train
        train_outputs, train_loss = ft_train(args, train_loader, model, criterion, optimizer, scaler, epoch)
        _, matrix_set = get_eval_metrics(train_outputs, args)
        log.info(f'Epoch:{epoch}, Train Accuracy:{matrix_set[1]}, f1:{matrix_set[0]}\n')  # , auc:{matrix_set[4]}

        # validation
        val_outputs, val_loss = validation(args, val_loader, model, criterion)
        _, matrix_set = get_eval_metrics(val_outputs, args)
        val_f1, val_acc = matrix_set[0], matrix_set[1]

        log.info(f'Epoch:{epoch}, Val Accuracy:{matrix_set[1]}, f1:{matrix_set[0]}\n')  # , auc:{matrix_set[4]}

        scheduler.step()
        log.info('lr:{}'.format(optimizer.param_groups[0]['lr']))

        if val_acc < best_val_acc:
            early_st -= 1
            print('early_st', early_st)
            if early_st <= 0:
                print('early_stop')
                break
        else:
            print('Save best weights')
            best_val_acc = val_acc

            import copy
            early_st = args.early_stop
            best_model_wts = copy.deepcopy(model.state_dict())
        if args.debug:
            break

    print('Test')
    model.load_state_dict(best_model_wts)

    test_outputs = test(args, test_loader, model)

    test_result, matrix_set = get_eval_metrics(test_outputs, args)
    log.info(
        f'test_f1:{matrix_set[0]}, test_acc:{matrix_set[1]}, test_precision:{matrix_set[2]}, test_recall:{matrix_set[3]}')


def linear_eval(args):
    train_loader, test_loader = dataset_loader.get_dataloader(args, mode='linear_eval')

    model = load_ft_model(args)

    for index, (name, named_param) in enumerate(model.named_parameters()):
        if 'fc' in name:
            print(named_param)
        else:
            named_param.requires_grad = False

    print('args.default_device', args.default_device)

    model = model.to(args.default_device)

    criterion = torch.nn.CrossEntropyLoss().to(args.default_device)

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, list(model.parameters())),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = get_scheduler(optimizer, 4e-8)

    # val
    best_test_acc = 0

    early_st = args.early_stop
    scaler = None
    if args.fp16:
        log.info('Use fp16 and loaded')
        scaler = torch.cuda.amp.GradScaler()
    if scaler:
        print(f'scaler: {scaler}')

    for epoch in range(args.ft_epoch):
        log.info('=' * 10 + f'Epoch:{epoch}' + '=' * 10)

        # train
        train_outputs, train_loss = ft_train(args, train_loader, model, criterion, optimizer, scaler, epoch)
        _, matrix_set = get_eval_metrics(train_outputs, args)
        log.info(f'Epoch:{epoch}, Train Accuracy:{matrix_set[1]}, f1:{matrix_set[0]}\n')

        # Test
        test_outputs, test_loss = validation(args, test_loader, model, criterion)
        _, matrix_set = get_eval_metrics(test_outputs, args)

        test_acc = matrix_set[1]
        test_f1 = matrix_set[0]
        log.info(f'Epoch:{epoch}, test Accuracy:{matrix_set[1]}, f1:{matrix_set[0]}\n')  # , auc:{matrix_set[4]}

        scheduler.step()
        log.info('lr:{}'.format(optimizer.param_groups[0]['lr']))

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            best_test_f1 = test_f1
            best_model_wts = copy.deepcopy(model.state_dict())

        if args.debug:
            break
    print('Test')
    model.load_state_dict(best_model_wts)

    test_outputs = test(args, test_loader, model)

    test_result, matrix_set = get_eval_metrics(test_outputs, args)

    log.info(
        f'test_f1:{matrix_set[0]}, test_acc:{matrix_set[1]}, test_precision:{matrix_set[2]}, test_recall:{matrix_set[3]}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--dataset_path', type=str, default=r'C:\Users\junjianli\Desktop\code\data\Kather_Multi_Class',
                        help='dataset')
    parser.add_argument('--dataset', type=str, default=r'NCT',
                        help='dataset')
    parser.add_argument('--only_fine_turning', action="store_true", help='only do fine turning')
    parser.add_argument('--only_linear_eval', action="store_true", help='only do linear evaluation')

    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--fp16', action='store_true', help='fp16 mode')
    parser.add_argument('--ft_epoch', type=int, default=100, help='ft_epoch')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--lr', type=float, default=0.01, help='lr')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
    parser.add_argument('--workers', type=int, default=4, help='workers')
    parser.add_argument('--early_stop', type=int, default=5, help='early_stop')

    parser.add_argument('--model_name', type=str, default='resnet18', help='model_name')
    parser.add_argument('--image_size', type=int, default=224, help='image_size')

    parser.add_argument('--labeled_train', type=float, default=0.01, help='labeled_train')
    parser.add_argument('--dataloader_num_workers', type=int, default=16, help='dataloader_num_workers')
    parser.add_argument('--model_path', type=str, default='', help='model_path')
    parser.add_argument('--eval_metrics', type=str, default='accuracy,f1,precision,recall', help='eval_metrics')

    parser.add_argument('--validation_split', type=float, default=0.2, help='validation_split')
    parser.add_argument('--default_device', type=str, default='cuda', help='default_device')
    parser.add_argument('--gpu_index', type=str, default='0', help='gpu_index')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.default_device = args.default_device + ':' + args.gpu_index

    log.info(f"设置 seed 为:  {args.seed}")
    seed_everything(seed=args.seed)

    if args.only_fine_turning and not args.only_linear_eval:
        log.info('Start fine-turning...')
        finetuning(args)
        log.info('Finish fine-turning...')
    if not args.only_fine_turning and args.only_linear_eval:
        log.info('Start linear evaluation...')
        linear_eval(args)
        log.info('Finish linear evaluation...')


if __name__ == '__main__':
    main()