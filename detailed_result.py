import re
import pandas as pd

# 读取日志文件
file_path = 'result/sage_Flickr_result.txt'
with open(file_path, 'r') as file:
    log_data = file.read()

# 定义正则表达式
experiment_pattern = r"Running experiment with Layer=(\d+), Batch=(\d+), Device=(\d+), Stage=(\d+)"
# rate_pattern = r"rate:\((\d\.\d+:\d\.\d+)\) (\w{3}) time:(\d+\.\d+)ms; (\w{3}) time:(\d+\.\d+)ms"
rate_pattern = r"rate:\((\d\.\d+):(\d\.\d+)\) (\w{3}) time:(\d+\.\d+)ms; (\w{3}) time:(\d+\.\d+)ms"
max_cycle_time_pattern = r"max_cycle time:(\d+\.\d+)ms"
single_device_pattern = r"device:(\w{3}) model:\w+ time:(\d+\.\d+)ms"

# 初始化存储提取数据的列表
records = []

# 逐个实验块进行提取和处理
for experiment_match in re.finditer(experiment_pattern, log_data):
    layer, batch, device, stage = experiment_match.groups()
    comb =  int(device)*2 + int(stage)
    start_idx = experiment_match.end()

    # 定位下一个实验块或文件末尾
    
    next_experiment_match = re.search(experiment_pattern, log_data[start_idx:])
    end_idx = start_idx + next_experiment_match.start() if next_experiment_match else len(log_data)

    # 提取当前实验块
    experiment_block = log_data[start_idx:end_idx]

    # 提取最大循环时间
    max_cycle_match = re.search(max_cycle_time_pattern, experiment_block)
    max_cycle_time = float(max_cycle_match.group(1)) if max_cycle_match else None
    single_device_match = re.search(single_device_pattern, experiment_block)
    if single_device_match:
        dev1, time1 = single_device_match.groups()
    for rate_match in re.finditer(rate_pattern, experiment_block):
        first_rate, second_rate, dev2, time2, dev3, time3 = rate_match.groups()
        Max_DP_time = max(float(time2), float(time3))
        Max_MIX_time = max(float(time1), float(time2), float(time3))
        first_rate_multiplied = int(float(first_rate) * 10)
        records.append([layer, batch, device, stage, comb, time1, first_rate_multiplied, float(time2), float(time3), max(float(time2), float(time3)), Max_MIX_time, max_cycle_time])

# 将数据转为DataFrame
df = pd.DataFrame(records, columns=["Layer", "Batch", "Device", "Stage", "Comb", "Time1", "Rate", "Time2", "Time3", "Max_DP_time", "Max_MIX_time", "Max_Cycle_Time"])

# 保存到CSV文件
df.to_csv("experiment_data.csv", index=False)


# import pandas as pd

# # 读取CSV文件
# df = pd.read_csv('experiment_data.csv')

# # 按Rate_Group累加 Max_Cycle_Time
# rate_grouped = df.groupby("Rate")["Max_MIX_time"].sum().reset_index()

# # 保存到新CSV文件
# rate_grouped.to_csv("rate_grouped_summary.csv", index=False)

# # 打印结果
# print(file_path)
# print("按Rate分组累加的Max_MIX_time结果：")
# print(rate_grouped)


# # 按Rate_Group累加 Max_Cycle_Time
# rate_grouped = df.groupby("Comb")["Max_MIX_time"].sum().reset_index()

# # 保存到新CSV文件
# rate_grouped.to_csv("rate_grouped_summary.csv", index=False)

# # 打印结果
# print("按Comb分组累加的Max_MIX_time结果：")
# print(rate_grouped)

# 读取CSV文件
print(file_path)
df = pd.read_csv('experiment_data.csv')

# 统计 Batch 总数
total_batches = df["Batch"].nunique()
print("Batch 总数：", total_batches)

# 按 Batch 分组，找到每个 Batch 中的 Max_MIX_time 最小值
min_values_per_batch = df.groupby("Batch")["Max_MIX_time"].min().reset_index()

# 计算所有 Batch 的最小值的累加
total_min_sum = min_values_per_batch["Max_MIX_time"].sum()

# # 打印每个 Batch 的最小值和累加总和
# print("每个 Batch 中 Max_MIX_time 的最小值：")
# print(min_values_per_batch)
print("\n所有 Batch 的最小 Max_MIX_time 累加和：", total_min_sum)


# 按Batch和Rate分组，找到每个Batch中每个Rate的最小Max_MIX_time
min_values_per_batch = df.loc[df.groupby(["Batch", "Comb"])["Max_MIX_time"].idxmin()]

# 然后在所有Batch中，对每个Rate组的最小值进行累加
sum_min_values_across_batches = min_values_per_batch.groupby("Comb")["Max_MIX_time"].sum().reset_index()
# 打印结果
print("所有Batch中，每个Comb的最小 Max_MIX_time 累加结果：")
print(sum_min_values_across_batches)


# 按Batch和Rate分组，找到每个Batch中每个Rate的最小Max_MIX_time
min_values_per_batch = df.loc[df.groupby(["Batch", "Rate"])["Max_MIX_time"].idxmin()]

# 然后在所有Batch中，对每个Rate组的最小值进行累加
sum_min_values_across_batches = min_values_per_batch.groupby("Rate")["Max_MIX_time"].sum().reset_index()
# 打印结果
print("所有Batch中，每个Rate的最小 Max_MIX_time 累加结果：")
print(sum_min_values_across_batches)

