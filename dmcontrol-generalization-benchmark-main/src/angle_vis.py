import os
import sys
import numpy as np
from datetime import datetime
import argparse
import json
import csv
from tqdm import tqdm
import torch

try:
    from env.wrappers import make_env
except ImportError:
    print("警告：无法导入env.wrappers，尝试使用dm_control直接创建环境")
    try:
        from dm_control import suite
        def make_env(domain_name, task_name, seed, episode_length, action_repeat, image_size, mode):
            return suite.load(domain_name, task_name, task_kwargs={'random': seed})
    except ImportError:
        print("错误：无法导入dm_control，请确保已安装")
        sys.exit(1)

class JointAngleAnalyzer:
    def __init__(self, domain_name="walker", task_name="walk", seed=42, 
                 episode_length=1000, action_repeat=1, image_size=84, output_dir="angle_data"):
        self.domain_name = domain_name
        self.task_name = task_name
        self.seed = seed
        self.episode_length = episode_length
        self.action_repeat = action_repeat
        self.image_size = image_size
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"数据将保存到目录: {os.path.abspath(output_dir)}")
        
        self.env = self.create_env()
        
        self.joint_angles = []
        self.joint_velocities = []
        self.rewards = []
        self.step_count = 0
        
        self.joint_info = self.analyze_env()
        print(f"环境分析完成：\n  关节数量：{len(self.joint_info['joint_names'])}\n  关节名称：{self.joint_info['joint_names']}")
        
    def create_env(self):
        try:
            env = make_env(
                domain_name=self.domain_name,
                task_name=self.task_name,
                seed=self.seed,
                episode_length=self.episode_length,
                action_repeat=self.action_repeat,
                image_size=self.image_size,
                mode='train'
            )
            return env
        except Exception as e:
            print(f"创建环境时出错：{str(e)}")
            sys.exit(1)
    
    def analyze_env(self):
        joint_info = {
            'joint_names': [],
            'joint_ranges': {},
            'joint_indices': {},
            'action_dim': 0,
            'obs_shape': None
        }
        
        observation = self.env.reset()
        
        action_spec = self.env.action_spec()
        joint_info['action_dim'] = action_spec.shape[0]
        
        if isinstance(observation, dict):
            if 'joints' in observation:
                joint_info['joint_names'] = [f'joint_{i}' for i in range(len(observation['joints']))]
                joint_info['obs_joint_key'] = 'joints'
            elif 'position' in observation:
                joint_info['joint_names'] = [f'position_{i}' for i in range(len(observation['position']))]
                joint_info['obs_joint_key'] = 'position'
            
            if 'velocities' in observation:
                joint_info['velocity_key'] = 'velocities'
            
        try:
            if hasattr(self.env, 'physics'):
                physics = self.env.physics
                if hasattr(physics, 'named'):
                    joint_names = []
                    qpos_names = physics.named.data.qpos.axes.row.names
                    for name in qpos_names:
                        if not name.startswith('root') and not (name.endswith('x') or name.endswith('y') or name.endswith('z')):
                            joint_names.append(name)
                    
                    if joint_names:
                        joint_info['joint_names'] = joint_names
                        joint_info['physics_available'] = True
        except:
            pass
        
        if not joint_info['joint_names']:
            joint_info['joint_names'] = [f'joint_{i}' for i in range(joint_info['action_dim'])]
        
        for i, name in enumerate(joint_info['joint_names']):
            joint_info['joint_indices'][name] = i
        
        return joint_info
    
    def get_joint_angles(self, observation, physics=None):
        joint_angles = {}
        joint_velocities = {}
        
        if isinstance(observation, dict):
            if 'joints' in observation and self.joint_info.get('obs_joint_key') == 'joints':
                for i, name in enumerate(self.joint_info['joint_names']):
                    if i < len(observation['joints']):
                        joint_angles[name] = float(observation['joints'][i])
            
            elif 'position' in observation and self.joint_info.get('obs_joint_key') == 'position':
                for i, name in enumerate(self.joint_info['joint_names']):
                    if i < len(observation['position']):
                        joint_angles[name] = float(observation['position'][i])
            
            if 'velocities' in observation and self.joint_info.get('velocity_key') == 'velocities':
                for i, name in enumerate(self.joint_info['joint_names']):
                    vel_name = f"{name}_vel"
                    if i < len(observation['velocities']):
                        joint_velocities[vel_name] = float(observation['velocities'][i])
        
        if not joint_angles and physics is not None and hasattr(physics, 'named'):
            try:
                for name in self.joint_info['joint_names']:
                    joint_angles[name] = float(physics.named.data.qpos[name])
                    
                    if hasattr(physics.named.data, 'qvel') and name in physics.named.data.qvel.axes.row.names:
                        vel_name = f"{name}_vel"
                        joint_velocities[vel_name] = float(physics.named.data.qvel[name])
            except:
                pass
        
        if not joint_angles:
            if hasattr(observation, 'shape') and len(observation.shape) > 0:
                n_joints = min(len(self.joint_info['joint_names']), observation.shape[0])
                for i in range(n_joints):
                    name = self.joint_info['joint_names'][i]
                    joint_angles[name] = float(observation[i])
        
        return joint_angles, joint_velocities
    
    def run_episode(self, max_steps=1000, agent=None, random_actions=True):
        print(f"开始运行回合，最大步数：{max_steps}")
        
        observation = self.env.reset()
        physics = self.env.physics if hasattr(self.env, 'physics') else None
        
        self.joint_angles = []
        self.joint_velocities = []
        self.rewards = []
        self.step_count = 0
        
        for step in tqdm(range(max_steps)):
            self.step_count = step
            
            if agent is not None:
                with torch.no_grad():
                    if hasattr(agent, 'select_action'):
                        action = agent.select_action(observation)
                    elif hasattr(agent, 'act'):
                        action = agent.act(observation, step, eval_mode=True)
                    else:
                        try:
                            action = agent(observation)
                        except:
                            print("警告: 模型接口不匹配，尝试直接调用")
                            action = agent(torch.FloatTensor(observation).to(agent.device)).cpu().numpy()
            elif random_actions:
                action_spec = self.env.action_spec()
                action = np.random.uniform(
                    action_spec.minimum, 
                    action_spec.maximum, 
                    size=action_spec.shape
                )
            else:
                action = np.zeros(self.joint_info['action_dim'])
            
            next_observation, reward, done, info = self.env.step(action)
            
            angles, velocities = self.get_joint_angles(next_observation, physics)
            
            self.joint_angles.append(angles)
            self.joint_velocities.append(velocities)
            self.rewards.append(reward)
            
            if step % 100 == 0 and step > 0:
                angle_str = ", ".join([f"{name}: {val:.2f}" for name, val in list(angles.items())[:3]])
                print(f"步数：{step}, 奖励：{reward:.2f}, 部分关节角度：{angle_str}...")
            
            if done:
                print(f"回合在步数 {step} 结束")
                break
            
            observation = next_observation
        
        print(f"回合结束，收集了 {len(self.joint_angles)} 个数据点")
        return self.joint_angles, self.joint_velocities, self.rewards
    
    def save_data(self, filename=None):
        if not self.joint_angles:
            print("没有数据可保存")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.domain_name}_{self.task_name}_{timestamp}"
        
        csv_path = os.path.join(self.output_dir, f"{filename}.csv")
        
        headers = ['step']
        for name in self.joint_info['joint_names']:
            headers.append(f"angle_{name}")
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            for i in range(len(self.joint_angles)):
                row = {'step': i}
                
                for name in self.joint_info['joint_names']:
                    if name in self.joint_angles[i]:
                        row[f"angle_{name}"] = self.joint_angles[i][name]
                    else:
                        row[f"angle_{name}"] = ''
                
                writer.writerow(row)
        
        print(f"关节角度数据已保存到：{os.path.abspath(csv_path)}")
        return csv_path

def load_agent(model_path):
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 {model_path}")
        return None
    
    try:
        print(f"加载模型: {model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = torch.load(model_path, map_location=device)
        
        if hasattr(agent, 'eval'):
            agent.eval()
            print("模型已设置为评估模式")
        
        model_type = agent.__class__.__name__ if hasattr(agent, '__class__') else type(agent).__name__
        print(f"成功加载模型，类型: {model_type}")
        
        return agent
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="记录DeepMind Control Suite环境中的关节角度")
    
    parser.add_argument('--domain', type=str, default='walker', help='环境领域，如walker, humanoid等')
    parser.add_argument('--task', type=str, default='walk', help='任务名称，如walk, run等')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--steps', type=int, default=1000, help='最大步数')
    
    parser.add_argument('--model', type=str, default=None, help='预训练模型路径')
    parser.add_argument('--random', action='store_true', help='使用随机动作(如果指定了模型，则忽略此选项)')
    
    parser.add_argument('--output', type=str, 
                       default='/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/angle_data', 
                       help='输出目录')
    parser.add_argument('--filename', type=str, default=None, help='输出文件名前缀(不包含扩展名)')
    
    args = parser.parse_args()
    
    analyzer = JointAngleAnalyzer(
        domain_name=args.domain,
        task_name=args.task,
        seed=args.seed,
        output_dir=args.output
    )
    
    agent = None
    use_random_actions = args.random
    
    if args.model:
        agent = load_agent(args.model)
        if agent:
            use_random_actions = False
            print(f"将使用加载的模型生成动作")
        else:
            print(f"模型加载失败，将使用随机动作")
            use_random_actions = True
    else:
        print(f"未指定模型，将使用随机动作")
    
    analyzer.run_episode(
        max_steps=args.steps, 
        agent=agent, 
        random_actions=use_random_actions
    )
    
    csv_path = analyzer.save_data(args.filename)
    
    print(f"分析完成，关节角度数据已保存到 {csv_path}")

if __name__ == "__main__":
    main()



'''
python angle_vis.py --domain walker --task walk --model /mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_walker_walk_20241224_111447/model/500000.pt --steps 1000 --output angle_data

'''