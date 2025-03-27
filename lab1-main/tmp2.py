import re
import sys

def compute_speedups(times):
    baseline = times[0]
    # 计算每个 kernel 的 speedup = baseline / kernel_time
    return [baseline / t for t in times]

def parse_file(filename):
    groups = {}  # 使用字典存储，key 为 group 名称，value 为该组所有 kernel 的时间列表
    current_group = None
    # 匹配 "Kernel X: Geometric mean elapsed time = NUMBER ns" 的正则
    kernel_pattern = re.compile(r"Kernel\s+\d+:\s+Geometric mean elapsed time\s*=\s*(\d+)\s+ns")

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except IOError as e:
        print("Error opening file:", e)
        sys.exit(1)

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # 如果该行全为数字，则认为是一个 group 的标识
        if stripped.isdigit():
            current_group = stripped
            groups[current_group] = []
        else:
            # 尝试匹配 kernel 行
            m = kernel_pattern.search(stripped)
            if m and current_group is not None:
                try:
                    num = int(m.group(1))
                    groups[current_group].append(num)
                except ValueError:
                    print(f"Warning: Cannot convert '{m.group(1)}' to int.")
            else:
                print(f"Warning: Unrecognized line format: {stripped}")
    return groups

def print_group_speedups(groups):
    for group_name, times in groups.items():
        if not times:
            print(f"Group {group_name}: No kernel data found.\n")
            continue
        speedups = compute_speedups(times)
        print(f"Group {group_name}:")
        for idx, (t, sp) in enumerate(zip(times, speedups), start=1):
            print(f"  Kernel {idx}: Average elapsed time = {t} ns, speedup = {sp:.2f}")
        print()

def main():
    if len(sys.argv) < 2:
        print("Usage: {} <tmp.txt>".format(sys.argv[0]))
        sys.exit(1)
    
    filename = sys.argv[1]
    groups = parse_file(filename)
    print_group_speedups(groups)

if __name__ == "__main__":
    main()
