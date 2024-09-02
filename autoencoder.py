import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, in_channels, shrink_factor):
        super(AutoEncoder, self).__init__()
        shrinked_channels = max(int(in_channels / shrink_factor), 1)
        print("shrinked_channels为：", shrinked_channels)
        # Use a multilayer encoder and apply regularization methods such as Dropout or BatchNorm to prevent overfitting
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, shrinked_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(shrinked_channels)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(shrinked_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels)
        )

    # x is the input data tensor and qbit is a parameter indicating the number of quantization bits
    def forward(self, x, qbit):
        code = self.encoder(x)
        if qbit is not None:
            code = self.quantize(code, qbit)
            code = self.dequantize(code, qbit)
        out = self.decoder(code)
        # print("out：",out)
        return out

    def quantize(self, x, qbit):
        # print("quantize!")
        self.max = x.max()
        self.min = x.min()
        return torch.round((2 ** qbit - 1) * (x - self.min) / (self.max - self.min) - 0.5)

    def dequantize(self, x, qbit):
        # print("dequantize!")
        return x * (self.max - self.min) / (2 ** qbit - 1) + self.min


class AutoencoderHook:
    def __init__(self, model, module, shrink_factor, qbit=None):
        hook = module.register_forward_pre_hook(self.channel_test_fn)  # add hook before module
        with torch.no_grad():
            model.eval()
            model(torch.zeros(1, 3, 224, 224).cuda())
        in_channels = self.shape[1]  # Get number of input channels
        hook.remove()

        self.ae = AutoEncoder(in_channels, shrink_factor).cuda()  # AutoEncoder forward()
        self.hook = module.register_forward_pre_hook(self.hook_fn)
        self.qbit = qbit

    # Define hook functions to be called during forward propagation and before the registered module is executed
    def hook_fn(self, module, input):
        self.input = input[0].detach()
        output = self.ae(input[0], self.qbit)
        self.output = output
        #print("module：",module)
        return output

    # Get Input Channel
    def channel_test_fn(self, module, input):
        self.shape = input[0].shape

    def parameters(self):
        return self.ae.parameters()

    def state_dict(self):
        return self.ae.state_dict()

    def load_state_dict(self, *args, **kwargs):
        self.ae.load_state_dict(*args, **kwargs)

    def train(self):
        self.ae.train()

    def eval(self):
        self.ae.eval()

    def close(self):
        self.hook.remove()


if __name__ == '__main__':
    from torchvision.models import resnet18, mobilenet_v2

    net = resnet18().cuda()
    hook = AutoencoderHook(net, net.layer1, 4)  # shrink factor=4

    in_channels = hook.shape[1]
    shrink_factor = 4
    shrinked_channels = max(int(in_channels / shrink_factor), 1)
    compression_rate = in_channels / shrinked_channels
    print(f"Number of input channels: {in_channels}")
    print(f"Number of channels after compression: {shrinked_channels}")
    print(f"Compression rate: {compression_rate}")

