import os
import warnings
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Events, Engine
from ignite.metrics import RunningAverage, Loss, ConfusionMatrix, IoU
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader

from sunnynet.datasets import DENSE, Normalize, Compose, RandomHorizontalFlip
from sunnynet.datasets.transforms import ToTensor
from sunnynet.model import SunnyNet
from sunnynet.utils import save

import datetime
import matplotlib.pyplot as plt
import numpy as np


def get_data_loaders(data_dir, batch_size, val_batch_size, num_workers):
    normalize = Normalize(mean=DENSE.mean(), std=DENSE.std())
    transforms = Compose([
        RandomHorizontalFlip(),
        ToTensor(),
        normalize
    ])

    val_transforms = Compose([
        ToTensor(),
        normalize
    ])

    train_loader = DataLoader(DENSE(root=data_dir, split='train', transform=transforms),
                              batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(DENSE(root=data_dir, split='val', transform=val_transforms),
                            batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(DENSE(root=data_dir, split='test', transform=val_transforms),
                            batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def run(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using GPU:", torch.cuda.get_device_name())

    num_classes = DENSE.num_classes()
    model = SunnyNet(num_classes, args.attention_type)

    device_count = torch.cuda.device_count()
    if device_count > 1:
        print("Using %d GPU(s)" % device_count)
        model = nn.DataParallel(model)
        args.batch_size = device_count * args.batch_size
        args.val_batch_size = device_count * args.val_batch_size

    model = model.to(device)

    train_loader, val_loader, test_loader = get_data_loaders(args.dataset_dir, args.batch_size, args.val_batch_size,
                                                args.num_workers)

    criterion = nn.CrossEntropyLoss(weight=DENSE.class_weights()).to(device)
    # optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=args.lr)
    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr)

    begin_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    output_dir = os.path.join(args.output_dir, 'model_' + args.attention_type + '_' + begin_time)

    if args.resume:
        if os.path.isfile(args.resume):
            if args.test == 'True':
                print("Loading model '{}'".format(args.resume))
                args.start_epoch = 0
                model.load_state_dict(torch.load(args.resume))
                print("Loaded model '{}'".format(args.resume))
            else:
                print("Loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['model'])
                # optimizer.load_state_dict(checkpoint['optimizer'])
                print("Loaded checkpoint '{}' (Epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint / model found at '{}'".format(args.resume))

    if args.lr_decay == 'True':
        scheduler = LambdaLR(optimizer, last_epoch=args.start_epoch, lr_lambda=lambda epoch: 1 / epoch)
    print("Using attention type:", args.attention_type)
    print("Using learning rate:", optimizer.param_groups[0]['initial_lr'])

    def _prepare_batch(batch, non_blocking=True):
        distance, reflectivity, target, x, y, z = batch

        return (convert_tensor(distance, device=device, non_blocking=non_blocking),
                convert_tensor(reflectivity, device=device, non_blocking=non_blocking),
                convert_tensor(target, device=device, non_blocking=non_blocking))

    def _update(engine, batch):
        model.train()

        if engine.state.iteration % engine.state.epoch_length == 1:
            print("The learning rate of NO.%d epoch：%.9f" % (engine.state.epoch, optimizer.param_groups[0]['lr']))
        if engine.state.iteration % args.grad_accum == 0:
            optimizer.zero_grad()
        distance, reflectivity, target = _prepare_batch(batch)
        pred = model(distance, reflectivity)
        loss = criterion(pred, target) / args.grad_accum
        loss.backward()
        if engine.state.iteration % args.grad_accum == 0:
            optimizer.step()
        if args.lr_decay == 'True':
            if engine.state.iteration % engine.state.epoch_length == 0:
                scheduler.step()

        return loss.item()

    trainer = Engine(_update)

    # attach running average metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    # attach progress bar
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=['loss'])

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            distance, reflectivity, target = _prepare_batch(batch)
            pred = model(distance, reflectivity)

            return pred, target

    evaluator = Engine(_inference)
    cm = ConfusionMatrix(num_classes)
    IoU(cm, ignore_index=0).attach(evaluator, 'IoU')
    Loss(criterion).attach(evaluator, 'loss')
    cm.attach(evaluator, 'cm')

    pbar2 = ProgressBar(persist=True, desc='Eval Epoch')
    pbar2.attach(evaluator)

    def _global_step_transform(engine, event_name):
        if trainer.state is not None:
            return trainer.state.iteration
        else:
            return 1

    tb_logger = TensorboardLogger(args.log_dir)
    tb_logger.attach(trainer,
                     log_handler=OutputHandler(tag='training',
                                               metric_names=['loss']),
                     event_name=Events.ITERATION_COMPLETED)

    tb_logger.attach(evaluator,
                     log_handler=OutputHandler(tag='validation',
                                               metric_names=['loss', 'IoU'],
                                               global_step_transform=_global_step_transform),
                     event_name=Events.EPOCH_COMPLETED)

    @trainer.on(Events.STARTED)
    def initialize(engine):
        if args.resume:
            engine.state.epoch = args.start_epoch

    @evaluator.on(Events.EPOCH_COMPLETED)
    def save_checkpoint(engine):
        epoch = trainer.state.epoch if trainer.state is not None else 1
        iou = engine.state.metrics['IoU'] * 100.0
        mean_iou = iou.mean()

        name = 'epoch{}_mIoU={:.1f}.pth'.format(epoch, mean_iou)
        file = {'model': model.state_dict(), 'epoch': epoch, 'optimizer': optimizer.state_dict(),
                'args': args}

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        save(file, output_dir, 'checkpoint_{}'.format(name))
        save(model.state_dict(), output_dir, 'model_{}'.format(name))

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        pbar.log_message("Start Validation - Epoch: [{}/{}]".format(engine.state.epoch, engine.state.max_epochs))
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        loss = metrics['loss']
        iou = metrics['IoU'] * 100.0
        mean_iou = iou.mean()

        iou_text = ', '.join(['{}: {:.1f}'.format(DENSE.classes[i + 1].name, v) for i, v in enumerate(iou.tolist())])
        pbar.log_message("Validation results - Epoch: [{}/{}]: Loss: {:.2e}\n IoU: {}\n mIoU: {:.1f}"
                         .format(engine.state.epoch, engine.state.max_epochs, loss, iou_text, mean_iou))

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            engine.terminate()
            warnings.warn("KeyboardInterrupt caught. Exiting gracefully.")

            name = 'epoch{}_exception.pth'.format(trainer.state.epoch)
            file = {'model': model.state_dict(), 'epoch': trainer.state.epoch, 'optimizer': optimizer.state_dict(),
                    'args': args}

            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            save(file, output_dir, 'checkpoint_{}'.format(name))
            save(model.state_dict(), output_dir, 'model_{}'.format(name))
        else:
            raise e

    if args.test == 'True':
        print("Start testing")
        evaluator.run(test_loader, max_epochs=1)
        metrics = evaluator.state.metrics
        loss = metrics['loss']
        iou = metrics['IoU'] * 100.0
        mean_iou = iou.mean()

        iou_text = ', '.join(['{}: {:.1f}'.format(DENSE.classes[i + 1].name, v) for i, v in enumerate(iou.tolist())])
        pbar.log_message("Test results - Loss: {:.2e}\n IoU: {}\n mIoU: {:.1f}"
                         .format(loss, iou_text, mean_iou))

        # 混淆矩阵绘制
        cm_show = metrics['cm'].numpy()
        cm_show = 100 * cm_show[1:, 1:] / cm_show[1:, 1:].sum(axis=1).reshape(3, -1)
        classes = ['clear', 'rain', 'fog']
        title = 'Normalized confusion matrix'
        fig, ax = plt.subplots()
        im = ax.imshow(cm_show, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm_show.shape[1]),
               yticks=np.arange(cm_show.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        ax.set_ylim(len(classes) - 0.5, -0.5)

        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = cm_show.max() / 2.
        for i in range(cm_show.shape[0]):
            for j in range(cm_show.shape[1]):
                ax.text(j, i, format(cm_show[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm_show[i, j] > thresh else "black")
        fig.tight_layout()
        plt.show()
        return

    print("Start training")
    trainer.run(train_loader, max_epochs=args.epochs)
    tb_logger.close()


if __name__ == '__main__':
    parser = ArgumentParser('SunnyNet with PyTorch')
    parser.add_argument('--attention-type', type=str, default='original',
                        help='input attention type: cbam, eca, senet, original')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='input batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=10,
                        help='input batch size for validation')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='number of workers')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--lr-decay', type=str, default='True',
                        help='learning rate decay')
    parser.add_argument('--seed', type=int, default=123,
                        help='manual seed')
    parser.add_argument('--output-dir', default='checkpoints',
                        help='directory to save model checkpoints')
    parser.add_argument('--resume', type=str, metavar='PATH',
                        help='path to latest checkpoint (train mode) (default: none) or model (test mode)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument("--dataset-dir", type=str, default="/media/luozhen/Blurryface SSD/数据集/点云语义分割/雨雾天气/cnn_denoising",
                        help="location of the dataset")
    parser.add_argument("--test", type=str, default='False',
                        help="test mode")
    parser.add_argument('--grad-accum', type=int, default=1,
                        help='grad accumulation')

    run(parser.parse_args())
