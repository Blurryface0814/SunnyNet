#!/home/luozhen/anaconda3/envs/weathernet/bin/python3.6

import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2
import torch
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header
import sys
import os
import time
import open3d as o3d
import torchsummary as summary


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from sunnynet.datasets import DENSE
from sunnynet.model import SunnyNet


COLOR_LABEL_MAPPING = {
    0: [0, 0, 0],
    100: [158, 158, 158],
    101: [0, 153, 153],
    102: [115, 0, 230],
}


class RosPublisher:
    def __init__(self, color_label_mapping=COLOR_LABEL_MAPPING):
        """initialize ros python api with bode 'RosPublisher' and set data format"""
        self.color_label_mapping = color_label_mapping
        # init ros_publisher
        self.rostime = rospy.Time.now()
        self.ros_rate = rospy.Rate(100)
        self.ros_publisher_denoised = rospy.Publisher('Denoised_points', PointCloud2, queue_size=1)
        self.ros_publisher_labeled = rospy.Publisher('labeled_points', PointCloud2, queue_size=1)

        # define data format
        self.sensorX_1 = None
        self.sensorY_1 = None
        self.sensorZ_1 = None
        self.points = None
        self.distance_m_1 = None
        self.intensity_1 = None
        self.labels_1 = None
        self.num_surplus = None
        self.frame_id = None
        self.point_size = None
        self.remove_zero = rospy.get_param('remove_zero')
        self.after_processing = rospy.get_param('after_processing')
        self.nb_points = rospy.get_param('nb_points')
        self.radius = rospy.get_param('radius')
        self.beta = rospy.get_param('beta')

    def get_rgb(self, labels):
        """returns color coding according to input labels """
        r = g = b = np.zeros_like(labels)
        for label_id, color in self.color_label_mapping.items():
            r = np.where(labels == label_id, color[0] / 255.0, r)
            g = np.where(labels == label_id, color[1] / 255.0, g)
            b = np.where(labels == label_id, color[2] / 255.0, b)
        return r, g, b

    def publish(self, model):
        """publish a frame point cloud """

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id

        # denoise a frame point cloud
        self.labels_1 = denoise(model, self.distance_m_1, self.intensity_1)
        keeped_idx = np.where(self.labels_1.flatten() <= 100)
        denoised_num = self.point_size - len(list(keeped_idx)[0])
        print("sunnynet denoised {}({:.2f}%) points".format(denoised_num, 100 * denoised_num / self.point_size))

        # no label
        nolabel = np.where(self.labels_1.flatten() == 0)
        print('point cloud which is no label:', nolabel[0].size)

        # remove zero
        if self.remove_zero:
            before_remove = keeped_idx[0].size
            not_zero_idx = np.where(self.distance_m_1[keeped_idx] != 0)
            keeped_idx = keeped_idx[0][not_zero_idx]
            zero_num = before_remove - keeped_idx.size
            print("removed %d point which distance is zero" % zero_num)
        else:
            zero_num = 0

        # after processing
        if self.after_processing == 'lror':
            print('using lror processing')

            start_processing = time.time()
            _, processing_idx = LROR(self.points[keeped_idx], nb_points=self.nb_points,
                                     radius=self.radius, beta=self.beta)
            keeped_idx = keeped_idx[processing_idx]
            end_processing = time.time()
            print('total point cloud number after lror: {}'.format(keeped_idx.size))
            print('lror speed time: {} ms'.format(round(end_processing - start_processing, 2) * 1000))

            denoised_num = self.point_size - len(keeped_idx) - zero_num
            print("sunnynet & lror denoised {}({:.2f}%) points".format(denoised_num,
                                                                         100 * denoised_num / self.point_size))
        elif self.after_processing == 'ror':
            pass
        elif self.after_processing == 'none':
            print('no after processing')
        print()

        # output labeled point cloud
        # http://wiki.ros.org/rviz/DisplayTypes/PointCloud
        r, g, b = self.get_rgb(self.labels_1.flatten())
        fields_labeled = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('distance', 12, PointField.FLOAT32, 1),
            PointField('intensity', 16, PointField.FLOAT32, 1),
            PointField('r', 20, PointField.FLOAT32, 1),
            PointField('g', 24, PointField.FLOAT32, 1),
            PointField('b', 28, PointField.FLOAT32, 1)
        ]

        points_labeled = list(zip(
            self.sensorX_1.flatten(),
            self.sensorY_1.flatten(),
            self.sensorZ_1.flatten(),
            self.distance_m_1.flatten(),
            self.intensity_1.flatten(),
            r, g, b
        ))

        # publish labeled points
        cloud_labeled = pc2.create_cloud(header, fields_labeled, points_labeled)
        self.ros_publisher_labeled.publish(cloud_labeled)

        # output denoised point cloud
        fields_denoised = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('distance', 12, PointField.FLOAT32, 1),
            PointField('intensity', 16, PointField.FLOAT32, 1),
            ]

        points_denoised = list(zip(
            self.sensorX_1.flatten()[keeped_idx],
            self.sensorY_1.flatten()[keeped_idx],
            self.sensorZ_1.flatten()[keeped_idx],
            self.distance_m_1.flatten()[keeped_idx],
            self.intensity_1.flatten()[keeped_idx],
        ))

        # publish denoised points
        cloud_denoised = pc2.create_cloud(header, fields_denoised, points_denoised)
        self.ros_publisher_denoised.publish(cloud_denoised)

    def load_pc(self, data):
        pc = ros_numpy.point_cloud2.pointcloud2_to_array(data)

        if len(pc.shape) == 1:
            # velodyne point cloud
            points = np.zeros((pc.shape[0], 3))
            points[:, 0] = pc['x']
            points[:, 1] = pc['y']
            points[:, 2] = pc['z']
            intensity = pc['intensity']

        elif len(pc.shape) == 2:
            # rslidar point cloud
            points = np.zeros((pc.shape[0], pc.shape[1], 3))
            points[:, :, 0] = pc['x']
            points[:, :, 1] = pc['y']
            points[:, :, 2] = pc['z']
            intensity = pc['intensity']

            points = points.reshape(-1, 3)
            intensity = intensity.reshape(-1)

        else:
            # unknown type of point cloud
            rospy.logerr("unknown type of point cloud, please change 'RosPublisher.load_pc' part")
            return

        self.distance_m_1 = np.sqrt(np.sum(points[:, :3] ** 2, axis=1))
        distance_num = self.distance_m_1.flatten().size

        num_surplus = distance_num % 32
        length = distance_num - num_surplus
        self.num_surplus = num_surplus
        self.sensorX_1 = points[:, 0][:length]
        self.sensorY_1 = points[:, 1][:length]
        self.sensorZ_1 = points[:, 2][:length]
        self.points = np.concatenate([self.sensorX_1.reshape(-1, 1),
                                      self.sensorY_1.reshape(-1, 1),
                                      self.sensorZ_1.reshape(-1, 1)], axis=1)
        self.intensity_1 = intensity[:length]
        self.distance_m_1 = self.distance_m_1[:length]
        self.frame_id = data.header.frame_id
        self.point_size = length
        print('input {} point cloud, use {} point cloud'.format(pc.shape[0], self.point_size))


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


def callback(data, args):
    model = args

    pub = RosPublisher()
    pub.load_pc(data)
    pub.publish(model)


def main():
    # get parameter
    model_path = rospy.get_param('model_path')
    input_topic = rospy.get_param('input_point_cloud_topic')
    attention_type = rospy.get_param('attention_type')
    summary_model = rospy.get_param('summary_model')

    # init model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = DENSE.num_classes()
    model = SunnyNet(num_classes, attention_type=attention_type)
    model = model.to(device)

    # Init and Load in model
    model.load_state_dict(torch.load(model_path))

    # summary model
    if summary_model:
        summary.summary(model, [(1, 32, 400), (1, 32, 400)])

    model.eval()

    # init ros node sub and spin up
    rospy.init_node("PC_Denoiser")
    rospy.Subscriber(input_topic, PointCloud2, callback, model, queue_size=1)
    rospy.spin()


if __name__ == '__main__':
    main()
