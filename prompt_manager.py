from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
import re

class PromptManager:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.chatanywhere.tech/v1"
        )
        self.last_predicted_steps = []  # 记录上一次预测的步骤
        self.last_environment_feedback = ""  # 记录上一次环境反馈
        self.cur_environment_feedback = ""  # 记录当前环境反馈

    def describe_image(self, image, voice_command):
        """
        处理图像并生成描述
        :param image: PIL 图像对象
        :return: 图片描述字符串
        """
        with BytesIO() as img_byte_array:
            image.save(img_byte_array, format="JPEG")
            image_base64 = base64.b64encode(img_byte_array.getvalue()).decode("utf-8")
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一个能够描述用于导航的图像的 AI 助手。关注与任务目标f{voice_command}相关的内容。"},
                {"role": "user", "content": [
                    {"type": "text", "text": "请描述这张图片的内容，注意其中物体在视野中的方向以及物体之间相互的位置关系。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]}
            ],
            max_tokens=250
        )

        return response.choices[0].message.content.strip()

    def analyze_environment(self, voice_command,scene_desc, front_obstacles_distance):
        """
        分析当前环境并推理环境变化，加入障碍物距离信息
        :param scene_desc: 当前场景描述
        :param obstacles_distance: 障碍物与机器人的距离信息
        :param previous_results: 上一轮执行结果
        :return: 环境反馈信息
        """
        messages = [
            {"role": "system", "content": (
                "你是一个分析环境并推理与任务目标相关信息的 AI 助手，帮助机器人理解当前环境，找到任务目标，"
                "并确认相应的前方左侧、正前方、前方右侧相应有哪些与任务相关的物体。"
                
            )},
            {"role": "user", "content": (
                f"当前任务: {voice_command}\n"
                f"当前场景: 机器人可以看到前方120度的场景，描述如下: {scene_desc}\n"
                f"障碍物距离信息: {front_obstacles_distance}\n"
                "请分析当前环境并推理与任务相关的信息"
                "- '左侧兴趣点': 前方左侧的重要物体（如 '桌子', '门'）\n"
                "- '正前方兴趣点': 正前方的重要物体\n"
                "- '右侧兴趣点': 前方右侧的重要物体\n"
                "- '目标可达性\n"
                "分析': 结合图像与距离信息，分析当前环境的可通行性，以及任务目标是否出现在正中间、足够近的地方并且有无障碍物遮挡？\n"
            )}
        ]
   
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        
        self.cur_environment_feedback = response.choices[0].message.content.strip()
        print(f"环境分析: {self.cur_environment_feedback}")
        return self.cur_environment_feedback

    def parse_instruction(self, step):
        """
        解析 GPT 返回的导航指令
        :param step: 一个导航步骤字典
        :return: 解析出的动作、角度、移动距离
        """
        action = step['action']
        turn_angle = step['turn_angle']
        move_distance = step['move_distance']
        
        if action == "turn left":
            return "turn left", turn_angle, 0
        elif action == "turn right":
            return "turn right", turn_angle, 0
        elif action == "go straight":
            return "go straight", 0, move_distance
        elif action == "finish":
            return "finish", 0, 0
        return "stop", 0, 0  # 默认返回停止
    
    def plan_navigation_steps(self, voice_command, task_summary, obstacles_distance):
        """
        生成导航步骤
        :param scene_desc: 当前场景描述
        :param voice_command: 语音任务指令
        :param topo_map: 拓扑地图信息
        :param obstacles_distance: 障碍物距离信息
        :return: 一个包含步骤的字典
        """
        environment_feedback = self.cur_environment_feedback
        last_environment_feedback = self.last_environment_feedback
        last_predicted_steps = self.last_predicted_steps  # 上一步预测的 2 个步骤
        last_step = last_predicted_steps[1] if len(last_predicted_steps) > 1 else "无"

        messages = [
            {"role": "system", "content": "你是一个机器人导航 AI 助手，需根据环境描述、任务指令以及实时障碍物距离生成合理的导航步骤。如果当前环境内没有目标而上一步场景信息内有目标，可以更多参考上一步环境与动作规划"},
            {"role": "user", "content": (
            f"任务目标: {voice_command}\n"
            f"这是历史任务完成程度总结:\n{task_summary}\n"
            f"上一轮你给出的路径规划（2步）：{last_predicted_steps}\n"
            f"第一步已执行，现在在当前环境：\n{environment_feedback}\n"
            f"障碍物距离：{obstacles_distance}\n\n"
            "你需要分两步完成当前导航决策：\n"
            "第一步：判断上一轮预测的第二步是否合理（仅考虑导航目标和障碍物距离）\n"
            f"预测的第二步: {last_step}\n"
            "如果合理，请直接将它作为当前执行步骤；如果不合理，请忽略并重新规划一个当前要执行的步骤。\n"
            "第二步：预测下一轮的动作（不需要立即执行，但应有导航连贯性）\n\n"
            "每个动作的格式为：\n动作, 转角角度, 前进距离\n"
            "例如：go straight, 0, 1.5 或 turn left, 60, 0\n"
            "可用动作：'go straight', 'turn left', 'turn right', 'finish'\n"
            "约束条件：\n"
            "- go straight 的前进距离 < 2m，且比正前方障碍物距离少 0.4m\n"
            "- 转角角度为 0-90°，角度单位为度，距离单位为米\n"
            "如果选择前进则转角角度应该为0，选择转弯则前进距离为0\n"
            "- 只输出两行动作指令，不包含任何其他内容\n"
            )}
        ]
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        
        # 解析返回的步骤
        steps = []
        predicted_steps = response.choices[0].message.content.strip().split('\n')
        print(f"生成的导航步骤: {predicted_steps}")
        for idx, step in enumerate(predicted_steps, start=1):
            # 解析每个步骤，确保每个步骤是一个字典
            step_parts = re.split(r",\s*", step.strip().replace("'", "").replace('"', ""))
            if len(step_parts) == 3:
                action, turn_angle, move_distance = step_parts
                steps.append({
                    "step_number": idx,
                    "action": action,
                    "turn_angle": int(turn_angle) if turn_angle.isdigit() else 0,
                    "move_distance": float(move_distance) if move_distance.replace('.', '', 1).isdigit() else 0
                })
        
        return {"steps": steps}


    def get_navigation_instruction(self, scene_desc,voice_command, task_summary, obstacles_distance):
        """
        获取导航指令
        :param scene_desc: 当前场景描述
        :param voice_command: 任务指令
        :param topo_map: 拓扑地图信息
        :param obstacles_distance: 障碍物距离信息
        :return: 一个导航指令字符串（动作、角度、移动距离）
        """
        #环境分析
        self.cur_environment_feedback = self.analyze_environment(voice_command, scene_desc, obstacles_distance)
        # 获取规划步骤
        result = self.plan_navigation_steps(voice_command, task_summary, obstacles_distance)
        steps = result["steps"]
        self.last_environment_feedback = self.cur_environment_feedback  # 更新环境反馈
        self.last_predicted_steps = steps  # 更新预测步骤
        # 返回第一个步骤的解析内容
        if steps:
            first_step = steps[0]
            return first_step['action'], first_step['turn_angle'], first_step['move_distance']
        
        return "stop", 0, 0  # 若没有步骤，返回停止指令
    
    def find_similar_nodes(current_scene_analysis, valid_nodes, chatgpt_client):
        scores = []
        for node in valid_nodes:
            history_scene = node["scene_analysis"]
            prompt = f"以下是两个环境描述，请给出它们的相似性分数（0-1），1 表示完全相同。\n\n环境 A：{current_scene_analysis}\n环境 B：{history_scene}\n相似性分数："
            similarity_score = chatgpt_client.get_response(prompt)  # 调用 ChatGPT
            scores.append((node, float(similarity_score)))

        # 按相似度降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:3]  # 取相似度最高的 3 个历史节点



    def update_task_summary(self, prev_summary, scene_desc, latest_action, latest_feedback):
        """
        使用大模型迭代更新任务完成情况总结。
        :param prev_summary: 上一步的任务完成总结
        :param scene_desc: 当前场景视觉描述
        :param latest_action: 当前执行的动作（如 "go straight 0.5m"）
        :param latest_feedback: 当前的环境反馈信息
        :return: 更新后的任务完成总结语句
        """
        messages = [
            {"role": "system", "content": (
                "你是一个机器人导航状态追踪助手，负责根据机器人当前执行的动作、环境描述与前一阶段总结，持续更新任务完成情况。"
                "每次总结都应该体现机器人是否在向目标接近，以及整体任务是否接近完成。"
                "注意不要重复冗余描述，保持总结简洁，逻辑清晰。"
            )},
            {"role": "user", "content": (
                f"上一步总结: {prev_summary}\n"
                f"当前视觉描述: {scene_desc}\n"
                f"执行的动作: {latest_action}\n"
                f"环境反馈信息: {latest_feedback}\n\n"
                "请基于以上信息，总结当前任务完成情况，并给出是否朝目标更进一步（例如：“机器人绕过障碍后更接近目标”）。"
                "输出你的总结"
            )}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        new_summary = response.choices[0].message.content.strip()
        print(f"更新后的任务总结: {new_summary}")
        return new_summary


    def should_record_node(self, scene_desc, environment_feedback,  action_history):
        """
        判断是否应记录当前场景为新的拓扑节点
        :param scene_desc: 当前场景的描述文本
        :return: True（记录为新节点）或 False（不记录）
        """
        flag = any("go straight" in action["action"] for action in action_history)
        if not flag:
            return False
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个机器人导航 AI，你需要判断当前场景是否需要记录为新的导航节点。\n"
                    "如果场景发生了显著变化，例如进入新的房间、走廊、路口、或者遇到新标志物，则应记录。\n"
                    "如果机器狗只是在上一个节点处转弯并没有发生位置上的移动，则不需要\n"
                    "只回答 'yes' 或 'no'。"
                )
            },
            {
                "role": "user",
                "content": f"场景描述: {scene_desc},环境分析：{environment_feedback}\n是否需要记录为新的导航节点？"
            }
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return response.choices[0].message.content.strip().lower() == "yes"

    
