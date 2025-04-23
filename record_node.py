import os
import time
import json
import shutil
import networkx as nx
import matplotlib.pyplot as plt

class NodeRecorder:
    def __init__(self, base_dir="recorded_nodes"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.last_node = None  # 上一个节点结构体
        self.pending_actions = []
        self.nodes = []
        self.edges = []

    def record_new_node(self, scene_desc, image_path, action_history, prev_node, obstacle_info, environment_feedback, current_position, heading_angle):
        cur_node = {
            "id": len(self.nodes),
            "scene_desc": scene_desc,
            "image_path": image_path,
            "action_history": action_history,
            "obstacle_info": obstacle_info,
            "environment_feedback": environment_feedback,
            "position": current_position,
            "heading_angle": heading_angle,
            "timestamp": time.strftime('%Y%m%d_%H%M%S')
        }

        # 创建节点文件夹
        node_folder = os.path.join(self.base_dir, f"node_{cur_node['id']}")
        os.makedirs(node_folder, exist_ok=True)  # 确保文件夹存在
        cur_node["folder"] = node_folder

        # 保存图像文件到节点文件夹（如果需要）
        if image_path:
            shutil.copy(image_path, os.path.join(node_folder, "image.png"))

        # 创建并保存元数据文件（metadata.json）
        metadata = {
            "id": cur_node["id"],
            "scene_desc": cur_node["scene_desc"],
            "timestamp": cur_node["timestamp"],
            "next_node": None  # 初始时没有下一个节点
        }
        with open(os.path.join(node_folder, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        self.nodes.append(cur_node)

        # 如果有前驱节点，记录边
        if prev_node is not None:
            edge = {
                "from": prev_node["id"],
                "to": cur_node["id"],
                "actions": action_history
            }
            self.edges.append(edge)

        return cur_node

    def record_action_to_last_node(self, action, turn_angle, move_distance):
        action_entry = {
            "action": action,
            "turn_angle": turn_angle,
            "move_distance": move_distance
        }
        self.pending_actions.append(action_entry)
        if self.last_node:
            # 实时写入最后节点的 action_history.txt
            action_file = os.path.join(self.last_node["folder"], "action_history.txt")
            with open(action_file, "a") as f:
                f.write(f"Action: {action}, Turn: {turn_angle}, Move: {move_distance}m\n")
        print(f"[→] Action added: {action}")

    def load_node_metadata(self, folder):
        path = os.path.join(self.base_dir, folder, "metadata.json")
        with open(path, "r") as f:
            return json.load(f)

    def update_next_node(self, prev_folder, next_folder):
        path = os.path.join(self.base_dir, prev_folder, "metadata.json")
        with open(path, "r") as f:
            metadata = json.load(f)
        metadata["next_node"] = next_folder
        with open(path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"[↑] Updated next_node of {prev_folder} to {next_folder}")

    def draw_topo_map(self, save_path="topo_map.png"):
        G = nx.DiGraph()
        # 添加节点
        for node in self.nodes:
            node_id = node["id"]
            desc = node.get("scene_desc", "")
            label = f"{node_id}\n{desc[:10]}..." if desc else str(node_id)
            G.add_node(node_id, label=label)

        # 添加边
        for edge in self.edges:
            from_id = edge["from"]
            to_id = edge["to"]
            actions = edge.get("actions", [])
            action_str = "\n".join([
                f"{a['action']} ({a['move_distance']}m)" for a in actions
            ])
            G.add_edge(from_id, to_id, label=action_str)

        # 布局设置（层级分布好看点）
        pos = nx.spring_layout(G, seed=42)

        # 绘图
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'),
                node_color='skyblue', node_size=1200, font_size=10, font_weight='bold', arrows=True)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title("Robot Navigation Topology")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"[✔] 拓扑图已保存至 {save_path}")
