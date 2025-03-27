import sys
import re
import math

def main():
    if len(sys.argv) < 2:
        print("Usage: {} <log_file>".format(sys.argv[0]))
        sys.exit(1)
    
    filename = sys.argv[1]
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except IOError as e:
        print("Error opening file:", e)
        sys.exit(1)

    # 按空行分割区块，每个区块代表一次测试
    blocks = []
    current_block = []
    for line in lines:
        stripped = line.strip()
        if stripped == "":
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            current_block.append(stripped)
    if current_block:
        blocks.append(current_block)

    if not blocks:
        print("No data found in log file.")
        sys.exit(1)
    
    # 使用正则匹配 "Elapsed time (Machine readable):" 后面的数字
    pattern = re.compile(r"Elapsed time \(Machine readable\):\s*(\d+)")
    
    # 假设每个区块内的行数一致，先取第一个区块的行数作为 kernel 数量
    num_kernels = len(blocks[0])
    kernel_times = [[] for _ in range(num_kernels)]
    
    # 遍历每个区块，将每一行对应的 kernel 时间提取出来
    for b_idx, block in enumerate(blocks, start=1):
        if len(block) != num_kernels:
            print(f"Warning: Block {b_idx} has {len(block)} lines, expected {num_kernels}. Skipping this block.")
            continue
        for idx, line in enumerate(block):
            match = pattern.search(line)
            if match:
                try:
                    num = int(match.group(1))
                    kernel_times[idx].append(num)
                except ValueError:
                    print(f"Warning: Unable to convert number in block {b_idx}, line {idx+1}.")
            else:
                print(f"Warning: No match found in block {b_idx}, line {idx+1}.")

    # 计算每个 kernel 的几何平均时间并打印
    for idx, times in enumerate(kernel_times, start=1):
        if times:
            # 计算几何平均数: exp( sum(log(x))/n )
            try:
                log_sum = sum(math.log(x) for x in times)
                geo_mean = math.exp(log_sum / len(times))
                print(f"Kernel {idx}: Geometric mean elapsed time = {geo_mean:.0f} ns (n={len(times)})")
            except ValueError as e:
                print(f"Kernel {idx}: Error computing geometric mean: {e}")
        else:
            print(f"Kernel {idx}: No data available.")

if __name__ == "__main__":
    main()
