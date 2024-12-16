import sys
import os

# 添加本地dm_control路径
current_dir = os.path.dirname(__file__)
dm_control_path = os.path.join(current_dir, 'env/dm_control')
sys.path.append(dm_control_path)

# 直接导入suite模块
from dm_control import suite

import pprint

def list_all_tasks():
    # 创建一个字典来存储所有domain和其对应的tasks
    all_tasks = {}
    
    try:
        # 直接读取suite目录下的所有.py文件（除了__init__.py）
        suite_dir = os.path.join(dm_control_path, 'dm_control/suite')
        domain_files = [f[:-3] for f in os.listdir(suite_dir) 
                       if f.endswith('.py') and f != '__init__.py']
        
        print("Available domains:", domain_files)
        
        # 对每个domain文件尝试获取其任务
        for domain in domain_files:
            try:
                domain_module = __import__(f'dm_control.suite.{domain}', fromlist=['SUITE'])
                if hasattr(domain_module, 'SUITE'):
                    all_tasks[domain] = list(domain_module.SUITE.keys())
            except Exception as e:
                print(f"Error loading domain {domain}: {e}")
        
    except Exception as e:
        print(f"Error accessing suite directory: {e}")
    
    return all_tasks

def format_shell_array(items):
    """格式化shell数组字符串"""
    return '("' + '" "'.join(items) + '")'

def print_tasks():
    print("\n=== DMControl可用任务列表 ===\n")
    
    tasks_dict = list_all_tasks()
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(tasks_dict)
    
    if tasks_dict:  # 只在有任务时打印以下内容
        # 计算总任务数
        total_tasks = sum(len(tasks) for tasks in tasks_dict.values())
        print(f"\n总共有 {len(tasks_dict)} 个domains和 {total_tasks} 个tasks")
        
        # 打印适合run_experiments.sh的格式
        print("\n=== 建议的run_experiments.sh任务配置 ===\n")
        
        # 收集所有任务和对应的domain
        all_tasks = []
        all_domains = []
        for domain, domain_tasks in tasks_dict.items():
            all_tasks.extend(domain_tasks)
            all_domains.extend([domain] * len(domain_tasks))
        
        # 打印tasks和domains数组
        print(f"tasks={format_shell_array(all_tasks)}")
        print(f"\ndomains={format_shell_array(all_domains)}")

if __name__ == "__main__":
    # 打印Python路径以便调试
    print("Python path:", sys.path)
    print("Looking for suite in:", dm_control_path)
    print_tasks()

    # python find_task.py