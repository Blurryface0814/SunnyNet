#!/home/luozhen/anaconda3/envs/weathernet/bin/python3.6

import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2
import cv2


class RosPublisher:
    def __init__(self):
        # init ros_publisher
        self.rostime = rospy.Time.now()
        self.ros_rate = rospy.Rate(100)

        # define data format
        self.distance_m_1 = None
        self.intensity_1 = None

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
        self.intensity_1 = intensity

    def trans2image(self, type_name):
        if type_name == 'distance':
            img = self.distance_m_1.flatten()
        if type_name == 'intensity':
            img = self.intensity_1.flatten()
        num = 0
        for idx, data in enumerate(img):
            if np.isnan(data):
                img[idx] = 0
                num += 1
        print(num)
        max = np.max(img, keepdims=False)
        img = np.int8(255 * img / max)
        img = img.reshape(32, -1)
        print(img.shape)
        print()
        img = cv2.applyColorMap(cv2.convertScaleAbs(img, alpha=1), cv2.COLORMAP_JET)
        # size = (400 * 5, 32 * 5)
        size = (img.shape[1] * 5, img.shape[0] * 5)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        # imgd = cv2.cvtColor(imgd, cv2.COLOR_GRAY2BGR)
        # cv2.imshow('test', imgd)
        # cv2.waitKey(0)
        # cv2.destroyWindow('test')
        if type_name == 'distance':
            cv2.imwrite("/home/luozhen/distance.jpg", img)
        if type_name == 'intensity':
            cv2.imwrite("/home/luozhen/intensity.jpg", img)


def callback(data):
    pub = RosPublisher()
    pub.load_pc(data)
    pub.trans2image('distance')
    pub.trans2image('intensity')


def main():
    # get parameter
    input_topic = rospy.get_param('input_point_cloud_topic')

    # init ros node sub and spin up
    rospy.init_node("lidar2img")
    rospy.Subscriber(input_topic, PointCloud2, callback, queue_size=1)
    rospy.spin()


if __name__ == '__main__':
    main()
