import torch
import time
import sys
import math
import os
import cv2
import numpy as np
from PIL import Image
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
from lidar_range_info import RangeInfoSubscriber
from prompt_manager import PromptManager  # 导入封装好的 PromptManager
from record_node import NodeRecorder  # 导入封装好的 RecordNode

# OpenAI API 配置
API_KEY = os.environ.get("OPENAI_API_KEY")  # 获取环境变量中的 API 密钥
prompt_manager = PromptManager(api_key=API_KEY)

# **保存当前图像**
def save_image(video_client, pics_dir):
    if not os.path.exists(pics_dir):
        os.makedirs(pics_dir)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_name = os.path.join(pics_dir, f"front_image_{timestamp}.jpg")

    code, data = video_client.GetImageSample()
    if code == 0:
        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if cv2.imwrite(image_name, image):
            return os.path.abspath(image_name)
    return None

# **初始化 Unitree 机器人**
def initialize_robot():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface")
        sys.exit(-1)

    print("WARNING: Ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    # 初始化 Channel 和 SportClient
    ChannelFactoryInitialize(0, sys.argv[1])
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    # 初始化视频客户端
    video_client = VideoClient()
    video_client.SetTimeout(3.0)
    video_client.Init()

    return sport_client, video_client

# **日志记录函数**
def log_decision_process(log_file, scene_desc, obstacle_info, reasoning_steps, action_decision):
    with open(log_file, "a") as f:
        f.write(f"{time.strftime('%Y%m%d_%H%M%S')} - Scene: {scene_desc}\n")
        f.write(f"Obstacle Info: {obstacle_info}\n")
        f.write("Reasoning Steps:\n")
        for step in reasoning_steps:
            f.write(f"  - {step}\n")
        f.write(f"Action Decision: {action_decision}\n")
        f.write("="*50 + "\n")

# **主函数**
def main():
    #初始化机器人初始位置与朝向
    cur_pos = {"x" : 0, "y" : 0}
    heading_angle = 0  # 假设初始朝向为0度
    sport_client, video_client = initialize_robot()
    node_recorder = NodeRecorder()  # 创建节点记录器实例
    # 障碍物距离
    lidar_subscriber = RangeInfoSubscriber()
    
    # 任务目标
    voice_command = "找到纸箱旁边的红色椅子"
    valid_nodes = []  # 记录有效节点
    last_node = None  # 记录最近的一个节点
    pending_actions = []  # 记录未成为新节点的动作
    #任务总结
    task_summary = "机器人处于初始位置，开始执行任务。"
    log_file = "navigation_log.txt"  # 设置日志文件

    while True:

        # 获取并保存当前图像
        sport_client.StopMove()  # 停止机器人运动
        time.sleep(0.5)  # 等待机器人停止
        image_path = save_image(video_client, "pics")
        if image_path is None:
            print("Error: Image not saved, skipping this iteration.")
            continue
        image = Image.open(image_path)
        
        # 调用 lidar_subscriber.receive_message() 获取障碍物距离
        front_obstacle_distance, left_obstacle_distance, right_obstacle_distance = lidar_subscriber.receive_message()
        obstacle_info = f"Front Distance: {front_obstacle_distance:.2f}"
        print(obstacle_info)

        # **使用 GPT-4o 获取场景描述**
        scene_desc = prompt_manager.describe_image(image, voice_command)
        print(f"Scene: {scene_desc}")
        #scene_desc = f"left part : {left_desc}\n center part : {center_desc}\n right part: {right_desc}"
        
        # **获取导航指令**
        # 将障碍物信息作为参数传递给 get_navigation_instruction
        action, turn_angle, move_distance = prompt_manager.get_navigation_instruction(scene_desc, voice_command, task_summary, front_obstacle_distance)
        # **记录推理过程**
        reasoning_steps = [
            f"Step 1: Analyzing environment description.{prompt_manager.cur_environment_feedback}",
            f"Step 2: Analyzing obstacles: {obstacle_info}",
            f"Step 3: Determining task goal: {voice_command}",
            f"Step 4: Deciding next action: {action}."
        ]
        action_decision = f"Action: {action}, Turn: {turn_angle}, Move: {move_distance}m"
        task_summary = prompt_manager.update_task_summary(
            task_summary,
            scene_desc,
            action_decision,
            prompt_manager.cur_environment_feedback
        )
        # **记录推理过程到文件**
        log_decision_process(log_file, scene_desc, obstacle_info, reasoning_steps, action_decision)

        if action:
            print(f" -> Navigation Decision: {action}, Turn: {turn_angle}, Move: {move_distance}m")

        # **执行相应动作**
        if action == "turn left":
            sport_client.Move(0, 0, math.radians(turn_angle))
            heading_angle += turn_angle
            if heading_angle >= 360:
                heading_angle -= 360
        elif action == "turn right":
            sport_client.Move(0, 0, math.radians(-turn_angle))
            heading_angle -= turn_angle
            if heading_angle < 0:
                heading_angle += 360
        elif action == "go straight":
            for i in range(5):
                sport_client.Move(move_distance / 4, 0, 0)  # 前进
                time.sleep(1) 
            cur_pos["x"] += move_distance * math.cos(math.radians(heading_angle))
            cur_pos["y"] += move_distance * math.sin(math.radians(heading_angle))
        elif action == "stop":
            sport_client.StopMove()
        elif action == "finish":
            print("Task Completed: Robot has reached the target location!")
            sport_client.Move(0.2, 0, 0)  
            sport_client.StopMove()
            break  # 退出循环
        else:
            print(f" -> Invalid navigation instruction received, skipping.")
            continue
        print(f"action: {action}, turn_angle: {turn_angle}, move_distance: {move_distance}")


# 判断是否是新节点
        if prompt_manager.should_record_node(scene_desc, prompt_manager.cur_environment_feedback,node_recorder.pending_actions):
            new_node = node_recorder.record_new_node(
                scene_desc=scene_desc,
                image_path=image_path,
                action_history=pending_actions,
                prev_node=last_node,
                obstacle_info=obstacle_info,
                environment_feedback=prompt_manager.cur_environment_feedback,
                current_position=cur_pos.copy(),
                heading_angle=heading_angle
            )

            last_node = new_node
            pending_actions.clear()
        else:
            node_recorder.record_action_to_last_node(
                action=action,
                turn_angle=turn_angle,
                move_distance=move_distance
            )
            pending_actions.append({
                "action": action,
                "turn_angle": turn_angle,
                "move_distance": move_distance
            })
        node_recorder.draw_topo_map(save_path="topo_map.jpg")

        # **记录到文件**
        with open("valid_nodes.txt", "a") as f:
            f.write(f"{time.strftime('%Y%m%d_%H%M%S')} - Scene: {prompt_manager.cur_environment_feedback} -> Actions: {pending_actions}\n")

        time.sleep(1)  # 控制执行速率

if __name__ == "__main__":
    main()
