import torch
import torchvision.transforms as transforms
from dataset import get_loader
from autoencoder import AutoencoderHook

import time
import pynvml

# Data Conversion
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

module = modules[args.point]
hook = AutoencoderHook(net, module, args.factor,8)  #++++++qbit=8

train_loader, test_loader = get_loader('CT_scan', args.batch_size)

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_power_usage():
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
    return power

total_time = 0.0
total_power = 0.0
sample_count = 0

for i, (images, _) in enumerate(test_loader):
    if i >= 1000:
        break

    start_time = time.time()

    # Perform feature extraction
    with torch.no_grad():
        features = resnet18_model(images)

    # Perform feature compression
    with torch.no_grad():
        encoded_features, _ = autoencoder(features)

    end_time = time.time()

    # Computational delay
    elapsed_time = end_time - start_time
    total_time += elapsed_time

    power_usage = get_power_usage()
    total_power += power_usage

    sample_count += 1

average_latency = total_time / sample_count
average_power = total_power / sample_count

print(f'average delay: {average_latency:.6f} s')
print(f'average power: {average_power:.6f} w')
