import torch
import torchvision.models


class PretrainedResNet18Encoder(torch.nn.Module):
    """
    Pretrained ResNet18 model from torchvision.
    """
    def __init__(self, out_features=6):
        super(PretrainedResNet18Encoder, self).__init__()
        self.name: str = 'PretrainedResNet18'
        self.model_ft = torchvision.models.resnet18(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(self.model_ft.children())[:-1])
        self.fc = torch.nn.Linear(512, out_features)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x