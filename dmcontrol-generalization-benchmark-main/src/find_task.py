import sys
import os

current_dir = os.path.dirname(__file__)
dm_control_path = os.path.join(current_dir, 'env/dm_control')
sys.path.append(dm_control_path)

from dm_control import suite
import pprint

def get_action_dim(domain, task):
    try:
        env = suite.load(domain, task)
        return env.action_spec().shape[0]
    except Exception as e:
        return f"Error: {str(e)}"

def list_all_tasks():
    all_tasks = {}
    
    try:
        suite_dir = os.path.join(dm_control_path, 'dm_control/suite')
        domain_files = [f[:-3] for f in os.listdir(suite_dir) 
                       if f.endswith('.py') and f != '__init__.py']
        
        print("Available domains:", domain_files)
        
        for domain in domain_files:
            try:
                domain_module = __import__(f'dm_control.suite.{domain}', fromlist=['SUITE'])
                if hasattr(domain_module, 'SUITE'):
                    tasks = list(domain_module.SUITE.keys())
                    tasks_with_dims = {}
                    for task in tasks:
                        action_dim = get_action_dim(domain, task)
                        tasks_with_dims[task] = action_dim
                    all_tasks[domain] = tasks_with_dims
            except Exception as e:
                print(f"Error loading domain {domain}: {e}")
        
    except Exception as e:
        print(f"Error accessing suite directory: {e}")
    
    return all_tasks

def print_tasks():
    print("\n=== DMControl可用任务列表（包含动作维度）===\n")
    
    tasks_dict = list_all_tasks()
    
    for domain, tasks in tasks_dict.items():
        print(f"\n{domain}:")
        print("-" * 40)
        for task, action_dim in tasks.items():
            print(f"  {task:<20} Action Dimensions: {action_dim}")
    
    if tasks_dict:
        total_tasks = sum(len(tasks) for tasks in tasks_dict.values())
        print(f"\n总共有 {len(tasks_dict)} 个domains和 {total_tasks} 个tasks")
        
        dim_groups = {}
        for domain, tasks in tasks_dict.items():
            for task, dim in tasks.items():
                if isinstance(dim, int):
                    if dim not in dim_groups:
                        dim_groups[dim] = []
                    dim_groups[dim].append((domain, task))
        
        print("\n=== 按动作维度分组 ===")
        for dim, tasks in sorted(dim_groups.items()):
            print(f"\n动作维度 {dim}:")
            for domain, task in sorted(tasks):
                print(f"  - {domain:<15} {task}")

if __name__ == "__main__":
    print("Python path:", sys.path)
    print("Looking for suite in:", dm_control_path)
    print_tasks()

    # python find_task.py