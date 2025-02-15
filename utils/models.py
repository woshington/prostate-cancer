import torch
import torch.nn as nn
import efficientnet_pytorch as efficientnet_model

class EfficientNet(nn.Module):
    def __init__(self, backbone, output_dimensions, weights_path=None):
        super(EfficientNet, self).__init__()
        self.efficient_net = efficientnet_model.EfficientNet.from_pretrained(backbone)
        self.efficient_net.load_state_dict(
            torch.load(weights_path, weights_only=True)
        )
        self.fully_connected = nn.Linear(self.efficient_net._fc.in_features, output_dimensions)
        self.efficient_net._fc = nn.Identity()

    def extract(self, inputs):
        return self.efficient_net(inputs)

    def forward(self, inputs):
        x = self.extract(inputs)
        x = self.fully_connected(x)

        return x
