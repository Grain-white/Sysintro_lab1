#!/usr/bin/env python3

# 定义各组数据
group1 = [41824452374, 6392511520, 7438081947, 42261327251, 51932737950, 49908968936]
group2 = [42048116274, 6656497315, 8242244959, 41108265414, 5025715691]
group3 = [41675226785, 13662670219, 17013778462, 942853161]

def compute_speedups(times):
    baseline = times[0]
    # 计算每个 kernel 的 speedup = baseline / kernel_time
    return [baseline / t for t in times]

def print_group_speedups(group_data, group_name):
    speedups = compute_speedups(group_data)
    print(f"{group_name}:")
    for idx, sp in enumerate(speedups, start=1):
        print(f"  Kernel {idx}: Average elapsed time = {group_data[idx-1]} ns, speedup = {sp:.2f}")
    print()

# 计算并输出每组数据的 speedup
print_group_speedups(group1, "Group 1")
print_group_speedups(group2, "Group 2")
print_group_speedups(group3, "Group 3")
