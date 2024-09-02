import re

log_file_path = '/home/zsq/works/MOCI-Contrast/CTScanresult/vgg11_MOCI/user-num-one/n=5-output.log'

# Initializing Variables
total_reward = 0
total_delay = 0
total_energy = 0
count = 0

# Regular Expression Patterns
pattern = re.compile(r'reward (-?\d+\.\d+), \((\d+\.\d+)s (\d+\.\d+)j\)')

# Read log files
with open(log_file_path, 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match:
            reward = float(match.group(1))
            delay = float(match.group(2))
            energy = float(match.group(3))

            total_reward += reward
            total_delay += delay
            total_energy += energy
            count += 1

# Calculating averages
if count > 0:
    avg_reward = total_reward / count
    avg_delay = total_delay / count
    avg_energy = total_energy / count

    print(f'Average Reward: {avg_reward:.4f}')
    print(f'Average Delay: {avg_delay:.4f} s')
    print(f'Average Energy: {avg_energy:.4f} j')
else:
    print('No matches found')
