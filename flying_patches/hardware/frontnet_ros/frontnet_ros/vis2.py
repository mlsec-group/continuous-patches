import numpy as np

import rclpy
from rclpy.node import Node

from crazyflie_interfaces.msg import LogDataGeneric
from geometry_msgs.msg import PoseStamped


class VisualizationNode(Node):

    def __init__(self):
        super().__init__('vis')

        self.cf = "cf18"

        self.publisher = self.create_publisher(
                PoseStamped, "{}/frontnet_targetpos_typed".format(self.cf), 10)

        self.subscription1 = self.create_subscription(
            LogDataGeneric,
            '{}/frontnet'.format(self.cf),
            self.frontnet_targetpos_callback,
            10)


    def frontnet_targetpos_callback(self, msg: LogDataGeneric):
        # the expected configuration is
        # vars: ["frontnet.targetx", "frontnet.targety", "frontnet.targetz", "frontnet.targetyaw"]

        # self.get_logger().info('I heard: "%s"' % msg)

        # publish as another pose to visualiz in rviz
        msg2 = PoseStamped()
        msg2.header = msg.header
        msg2.pose.position.x = msg.values[0]
        msg2.pose.position.y = msg.values[1]
        msg2.pose.position.z = msg.values[2]
        msg2.pose.orientation.x = 0.0
        msg2.pose.orientation.y = 0.0
        msg2.pose.orientation.z = 0.0
        msg2.pose.orientation.w = 1.0
        self.publisher.publish(msg2)


def main(args=None):
    rclpy.init(args=args)

    node = VisualizationNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()