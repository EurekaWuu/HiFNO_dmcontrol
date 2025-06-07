from dm_control import suite


print("检查BENCHMARKING任务集:")
benchmarking_domains = {domain for domain, _ in suite.BENCHMARKING}
print("BENCHMARKING domains:", benchmarking_domains)

print("\n检查EXTRA任务集:")
if hasattr(suite, 'EXTRA'):
    extra_domains = {domain for domain, _ in suite.EXTRA}
    print("EXTRA domains:", extra_domains)
else:
    print("suite没有EXTRA属性")

print("\n检查ALL_TASKS任务集:")
all_domains = {domain for domain, _ in suite.ALL_TASKS}
print("ALL_TASKS domains:", all_domains)


print("\n直接加载dog任务:")
try:
    env = suite.load('dog', 'run')
    print("dog_run任务可用！")
    print("动作空间维度:", env.action_spec().shape[0])
except Exception as e:
    print(f"dog_run任务不可用: {str(e)}")