import time
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PointStamped_
from unitree_sdk2py.go2.sport.sport_client import SportClient

TOPIC_RANGE_INFO = "rt/utlidar/range_info"

class RangeInfoSubscriber:
    def __init__(self):
        # 订阅 range_info 话题
        self.subscriber = ChannelSubscriber(TOPIC_RANGE_INFO, PointStamped_)
        self.subscriber.Init()

    def receive_message(self):
        """ 读取 range_info 话题数据并返回三个方向的障碍物距离 """
        while True:
            msg = self.subscriber.Read(1000)  # 超时时间 1000ms
            if msg is not None:
                front_range = msg.point.x
                left_range = msg.point.y
                right_range = msg.point.z
                return front_range, left_range, right_range  # 返回三个方向的障碍物距离
            time.sleep(0.1)  # 避免过快循环

if __name__ == '__main__':
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()
    
    subscriber = RangeInfoSubscriber()
    
    while True:
        front, left, right = subscriber.receive_message()
        print(f"Front Distance: {front:.2f} m, Left Distance: {left:.2f} m, Right Distance: {right:.2f} m")
        time.sleep(1)  # 每秒获取一次数据