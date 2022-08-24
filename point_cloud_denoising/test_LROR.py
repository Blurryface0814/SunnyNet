#!/home/luozhen/anaconda3/envs/weathernet/bin/python3.6

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import time
import open3d as o3d
import glob
import h5py
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from argparse import ArgumentParser


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from sunnynet.datasets import DENSE
from sunnynet.datasets.transforms import RosToTensor, RosNormalize, RosCompose, RosRandomHorizontalFlip
from sunnynet.model import SunnyNet


class Tester:
    def __init__(self, after_processing='lror', remove_zero='False',
                 nb_points=2, radius=0.2, beta=1.0):
        # define data format
        self.sensorX_1 = None
        self.sensorY_1 = None
        self.sensorZ_1 = None
        self.points = None
        self.distance_m_1 = None
        self.intensity_1 = None
        self.labels_1 = None
        self.target_1 = None
        self.points_size = None
        self.points_keep = None
        self.num_surplus = None
        self.remove_zero = remove_zero
        self.after_processing = after_processing
        self.cm_sunnynet = np.zeros((3, 3))
        self.cm_processing = np.zeros((3, 3))
        self.nb_points = nb_points
        self.radius = radius
        self.beta = beta

    def test(self, model):
        """test point cloud """
        # denoise a frame point cloud
        self.labels_1 = denoise(model, self.distance_m_1, self.intensity_1)
        keeped_idx = np.where(self.labels_1.flatten() <= 100)
        keeped_idx_net = keeped_idx
        denoised_num = self.points_size - len(list(keeped_idx)[0])
        print("sunnynet denoised {}({:.2f}%) points".format(denoised_num, 100 * denoised_num / self.points_size))

        # remap labels
        target_fog = np.where(self.target_1 == 102)
        label_fog = np.where(self.labels_1 == 102)

        self.target_1[target_fog] = 101
        self.labels_1[label_fog] = 101

        self.cm_sunnynet = self.cm_sunnynet + confusion_matrix(self.target_1.flatten(),
                                                                   self.labels_1.flatten(),
                                                                   labels=[0, 100, 101])

        # after processing
        if self.after_processing == 'lror':
            print('using lror processing')
            start_processing = time.time()
            _, processing_idx = LROR(self.points[keeped_idx[0]], nb_points=self.nb_points,
                                     radius=self.radius, beta=self.beta)
            keeped_idx = keeped_idx[0][processing_idx]
            end_processing = time.time()
            print('lror speed time: {} ms'.format(round(end_processing - start_processing, 2) * 1000))

        elif self.after_processing == 'none':
            print('no after processing')
        print()

        diff_idx = np.setdiff1d(keeped_idx_net[0], keeped_idx, assume_unique=True)
        self.labels_1 = np.array(self.labels_1).flatten()
        self.labels_1[diff_idx] = 101

        self.cm_processing = self.cm_processing + confusion_matrix(self.target_1.flatten(),
                                                                   self.labels_1.flatten(),
                                                                   labels=[0, 100, 101])

    def load_hdf5_file(self, filename):
        """
        load one single hdf5 file with point cloud data
        """
        with h5py.File(filename, "r", driver='core') as hdf5:
            # for channel in self.channels:
            self.sensorX_1 = hdf5.get('sensorX_1')[()]
            self.sensorY_1 = hdf5.get('sensorY_1')[()]
            self.sensorZ_1 = hdf5.get('sensorZ_1')[()]
            self.distance_m_1 = hdf5.get('distance_m_1')[()]
            self.intensity_1 = hdf5.get('intensity_1')[()]
            self.target_1 = hdf5.get('labels_1')[()]
            self.points_size = self.sensorX_1.flatten().size

        # remove distance == 0 point cloud
        if self.remove_zero == "True":
            orignal_num = self.distance_m_1.flatten().size
            not_zero_idx = np.where(self.distance_m_1.flatten() != 0)
            num_surplus = not_zero_idx[0].size % 32
            length = not_zero_idx[0].size - num_surplus
            self.num_surplus = num_surplus
            self.sensorX_1 = self.sensorX_1.flatten()[not_zero_idx][:length]
            self.sensorY_1 = self.sensorY_1.flatten()[not_zero_idx][:length]
            self.sensorZ_1 = self.sensorZ_1.flatten()[not_zero_idx][:length]
            self.distance_m_1 = self.distance_m_1.flatten()[not_zero_idx][:length]
            self.intensity_1 = self.intensity_1.flatten()[not_zero_idx][:length]
            self.target_1 = self.target_1.flatten()[not_zero_idx][:length]
            self.points_size = length
            print("removed %d point which distance is zero" % (orignal_num - not_zero_idx[0].size))
            print('input {} point cloud, use {} point cloud'.format(orignal_num, self.points_size))
        else:
            pass
        self.points = np.concatenate([self.sensorX_1.reshape(-1, 1),
                                      self.sensorY_1.reshape(-1, 1),
                                      self.sensorZ_1.reshape(-1, 1)], axis=1)

    # plot confusion matrix
    def plot_confusion_matrix(self, cmap=plt.cm.Blues):
        # compute confusion matrix
        cm_sunnynet = self.cm_sunnynet
        cm_processing = self.cm_processing
        cm_sunnynet = 100 * cm_sunnynet[1:, 1:] / cm_sunnynet[1:, 1:].sum(axis=1).reshape(2, -1)
        cm_processing = 100 * cm_processing[1:, 1:] / cm_processing[1:, 1:].sum(axis=1).reshape(2, -1)

        # compute iou & miou
        iou_wea, miou_wea = compute_iou(cm_sunnynet)
        iou, miou = compute_iou(cm_processing)
        print('sunnynet iou')
        print('iou: clear {:.2f}, rain&fog {:.2f}'.format(100 * iou_wea[0], 100 * iou_wea[1]))
        print('miou: {:.2f}'.format(100 * miou_wea))
        print('after processing iou')
        print('iou: clear {:.2f}, rain&fog {:.2f}'.format(100 * iou[0], 100 * iou[1]))
        print('miou: {:.2f}'.format(100 * miou))

        # set param
        classes = ['clear', 'rain & fog']

        if self.remove_zero == "True":
            remove_zero = 'remove zero'
        else:
            remove_zero = 'not remove zero'

        title_wea = 'no after processing, {}\n' \
                    'iou: clear {:.2f}, rain&fog {:.2f}, miou: {:.2f}\n' \
                    .format(remove_zero, 100 * iou_wea[0], 100 * iou_wea[1], 100 * miou_wea)

        if self.after_processing == 'none':
            title = 'no after processing, {}\n' \
                    'iou: clear {:.2f}, rain&fog {:.2f}, miou: {:.2f}\n' \
                    .format(remove_zero, 100 * iou[0], 100 * iou[1], 100 * miou)
        else:
            if self.after_processing == 'lror':
                nb = 'nb_points'
                nb_num = self.nb_points
                param = 'radius'
                param_num = self.radius
            title = '{}, {}\n' \
                    'iou: clear {:.2f}, rain&fog {:.2f}, miou: {:.2f}\n' \
                    '{}: {}, {}: {}'\
                    .format(self.after_processing, remove_zero,
                            100 * iou[0], 100 * iou[1], 100 * miou,
                            nb, nb_num, param, param_num)

        # plot confusion matrix
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        im_wea = ax[0].imshow(cm_sunnynet, interpolation='nearest', cmap=cmap)
        im = ax[1].imshow(cm_processing, interpolation='nearest', cmap=cmap)
        ax[0].figure.colorbar(im_wea, ax=ax[0])
        ax[1].figure.colorbar(im, ax=ax[1])
        # We want to show all ticks...
        ax[0].set(xticks=np.arange(cm_sunnynet.shape[1]),
                     yticks=np.arange(cm_sunnynet.shape[0]),
                     # ... and label them with the respective list entries
                     xticklabels=classes, yticklabels=classes,
                     title=title_wea,
                     ylabel='True label',
                     xlabel='Predicted label')
        ax[1].set(xticks=np.arange(cm_processing.shape[1]),
                     yticks=np.arange(cm_processing.shape[0]),
                     # ... and label them with the respective list entries
                     xticklabels=classes, yticklabels=classes,
                     title=title,
                     ylabel='True label',
                     xlabel='Predicted label')

        ax[0].set_ylim(len(classes) - 0.5, -0.5)
        ax[1].set_ylim(len(classes) - 0.5, -0.5)

        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh_wea = cm_sunnynet.max() / 2.
        for i in range(cm_sunnynet.shape[0]):
            for j in range(cm_sunnynet.shape[1]):
                ax[0].text(j, i, format(cm_sunnynet[i, j], fmt),
                              ha="center", va="center",
                              color="white" if cm_sunnynet[i, j] > thresh_wea else "black")

        thresh = cm_processing.max() / 2.
        for i in range(cm_processing.shape[0]):
            for j in range(cm_processing.shape[1]):
                ax[1].text(j, i, format(cm_processing[i, j], fmt),
                              ha="center", va="center",
                              color="white" if cm_processing[i, j] > thresh else "black")
        fig.tight_layout()
        plt.show()


def LROR(points, nb_points=2, radius=0.2, beta=1.0):
    # LROR
    distance_xy = np.sqrt(np.sum(points[:, :2] ** 2, axis=1))
    processing_area = beta * np.mean(distance_xy.flatten())  # 单位：m
    processing_in = np.where(distance_xy.flatten() <= processing_area)
    processing_out = np.where(distance_xy.flatten() > processing_area)
    print('processing_area: {} m'.format(processing_area))
    print('point cloud number using lror: {}'.format(processing_in[0].size))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[processing_in[0]])
    points_denoised, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)  # radius单位：m
    print('point cloud number after lror: {}'.format(len(ind)))

    keeped_idx = np.concatenate([processing_in[0][ind], processing_out[0]], axis=0)
    return points[keeped_idx], keeped_idx


def compute_iou(cm):
    intersection = np.diag(cm)  # 交集
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)  # 并集
    iou = intersection / union  # 交并比，即IoU
    miou = np.mean(iou)  # 计算MIoU
    return iou, miou


def normalize(inp, mean, std):
    mean = torch.tensor(mean, dtype=inp.dtype, device=inp.device)
    std = torch.tensor(std, dtype=inp.dtype, device=inp.device)
    return (inp - mean) / std


def denoise(model, distance_1, intensity_1):
    distance_1 = distance_1.reshape(32, -1)
    intensity_1 = intensity_1.reshape(32, -1)

    distance = torch.as_tensor(distance_1.astype(np.float32, copy=True)).contiguous()
    intensity = torch.as_tensor(intensity_1.astype(np.float32, copy=True)).contiguous()

    # normalize the point cloud data
    mean = DENSE.mean()
    std = DENSE.std()
    distance = normalize(distance, mean[0], std[0])
    intensity = normalize(intensity, mean[1], std[1])

    distance = distance.view(1, 1, 32, -1)
    intensity = intensity.view(1, 1, 32, -1)

    start = time.time()
    # Get predictions
    with torch.no_grad():
        # 加torch.cuda.synchronize()是为了时间同步，从而查找出最花时间的步骤
        torch.cuda.synchronize()
        pred = model(distance.cuda(), intensity.cuda())
        torch.cuda.synchronize()
    end = time.time()
    print('predict speed time: ', round(end - start, 2) * 1000, 'ms')

    # choose argmax prediction for each point
    pred = torch.argmax(pred, dim=1, keepdim=True)

    labels = torch.squeeze(pred).cpu().numpy()
    label_dict = {0: 0, 1: 100, 2: 101, 3: 102}
    labels = np.vectorize(label_dict.get)(labels)

    return labels


def main(args):
    # init PCD model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = DENSE.num_classes()
    model = SunnyNet(num_classes, args.attention_type)
    model = model.to(device)

    # get parameter
    model_path = args.model_path
    path = args.dataset_dir

    # Init and Load in model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    tester = Tester(after_processing=args.after_processing,
                    remove_zero=args.remove_zero,
                    nb_points=args.nb_points,
                    radius=args.radius,
                    beta=args.beta)

    # get all files inside the defined dir
    files = sorted(glob.glob(path + '*/*.hdf5'))
    print('Directory {} contains are {} hdf5-files'.format(path, len(files)))
    if len(files) == 0:
        print('Please check the input dir {}. Could not find any hdf5-file'.format(path))
    else:
        print('Start testing...')
        for frame, file in enumerate(files, start=1):
            print('{}/{} : {}'.format(frame, len(files), file))

            # load file
            tester.load_hdf5_file(file)

            # publish point cloud
            tester.test(model)
        tester.plot_confusion_matrix()


if __name__ == '__main__':
    parser = ArgumentParser('SunnyNet Test')
    parser.add_argument('--attention-type', type=str, default='original',
                        help='input attention type: cbam, eca, senet, original')
    parser.add_argument('--model-path', type=str, default="./checkpoints/model_eca_2022-08-24_15:30:56/model_epoch3_mIoU=89.7.pth",
                        help='path to model')
    parser.add_argument("--dataset-dir", type=str,
                        default="/media/luozhen/Blurryface SSD/数据集/点云语义分割/雨雾天气/cnn_denoising/test_01/",
                        help="location of the dataset")
    parser.add_argument("--after-processing", type=str, default="lror",
                        help="after processing type: lror, none")
    parser.add_argument("--remove-zero", type=str, default='False',
                        help="remove zero or not")
    parser.add_argument("--nb-points", type=int, default=2,
                        help="lror nb_points number")
    parser.add_argument("--radius", type=float, default=0.2,
                        help="lror radius: m")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="lror beta")
    main(parser.parse_args())
