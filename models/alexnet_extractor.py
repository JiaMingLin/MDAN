import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNetExtractor(nn.Module):

    def __init__(self):
        super(AlexNetExtractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.output_feature_len = 256 * 6 * 6
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        return x


def alexnet_extractor(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    extractor = AlexNetExtractor(**kwargs)
    if pretrained:
        alexnet_state_dict = model_zoo.load_url(model_urls['alexnet'])
        extractor_state_dict = extractor.state_dict()
        # we only retain the feature extractor parameters
        for k in alexnet_state_dict.keys():
            if k in extractor_state_dict.keys() and k.startswith('features'):
                extractor_state_dict[k] = alexnet_state_dict[k]
        
        extractor.load_state_dict(extractor_state_dict)
    return extractor

def test():
    import torch
    extractor = alexnet_extractor(pretrained=True)
    img = torch.rand(1,3,448,448)
    output = extractor(img)
    print(output.size())

if __name__ == '__main__':
    test()