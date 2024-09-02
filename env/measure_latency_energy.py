import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import time

model_path = "/home/zsq/works/MOCI/result/model_train_resnet18/test_latency_energy/CTScanckp.pt"  # 替换成你的模型文件路径
model = resnet18()

# Load model parameters, ignore mismatched layers
state_dict = torch.load(model_path)
state_dict = {k: v for k, v in state_dict.items() if
              k in model.state_dict() and v.size() == model.state_dict()[k].size()}
model.load_state_dict(state_dict, strict=False)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_path = "/home/zsq/works/MOCI/result/model_train_resnet18/test_latency_energy/image_0002.png"
img = Image.open(img_path).convert("RGB")
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)


# Four stages of model definition
class ResNet18_Segments(torch.nn.Module):
    def __init__(self, original_model):
        super(ResNet18_Segments, self).__init__()
        self.segment1 = torch.nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool
        )
        self.segment2 = torch.nn.Sequential(
            original_model.layer1
        )
        self.segment3 = torch.nn.Sequential(
            original_model.layer2,
            original_model.layer3
        )
        self.segment4 = torch.nn.Sequential(
            original_model.layer4,
            original_model.avgpool
        )
        self.fc = original_model.fc

    def forward(self, x):
        x1 = self.segment1(x)
        x2 = self.segment2(x1)
        x3 = self.segment3(x2)
        x4 = self.segment4(x3)
        x4 = torch.flatten(x4, 1)
        out = self.fc(x4)
        return x1, x2, x3, x4, out


# Segmentation Model
segmented_model = ResNet18_Segments(model)

def measure_time_and_energy(model, input_data):
    segment_times = []
    segment_energies = []
    total_energy = 0
    prev_output = input_data

    for i in range(5):
        start_time = time.time()
        if i == 0:
            with torch.no_grad():
                output = model.segment1(prev_output)
        elif i == 1:
            with torch.no_grad():
                output = model.segment2(prev_output)
        elif i == 2:
            with torch.no_grad():
                output = model.segment3(prev_output)
        elif i == 3:
            with torch.no_grad():
                output = model.segment4(prev_output)
        else:
            with torch.no_grad():
                output = model.fc(torch.flatten(prev_output, 1))

        end_time = time.time()
        segment_time = end_time - start_time
        segment_times.append(segment_time)

        # Simulated power consumption, assuming 1 watt per second consumption
        segment_energy = segment_time * 1
        segment_energies.append(segment_energy)
        total_energy += segment_energy

        prev_output = output

    return segment_times, segment_energies

times, energies = measure_time_and_energy(segmented_model, batch_t)

for i in range(5):
    print(f"Segment {i + 1} - Time: {times[i]} seconds, Energy: {energies[i]} joules")

print(f"Total Energy Consumption: {sum(energies)} joules")

