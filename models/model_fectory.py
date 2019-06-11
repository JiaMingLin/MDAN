import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from .alexnet_extractor import alexnet_extractor
from .vgg16_extractor import vgg16_extractor
from .resnet50_extractor import resnet50_extractor

class GradientReversalLayer(torch.autograd.Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """
    def forward(self, inputs):
        return inputs

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = -grad_input
        return grad_input

class MDANet(nn.Module):
    """
    Multi-layer perceptron with adversarial regularizer by domain classification.
    """
    def __init__(self, class_num, domain_num, extractor):
        super(MDANet, self).__init__()
        
        # alexnet as feature extractor
        self.feature_extractor = extractor
        # image classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.feature_extractor.output_feature_len, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, class_num),
        )
        
        # Parameter of the domain classification layer, multiple sources single target domain adaptation.
        self.domain_classifier = nn.ModuleList([nn.Linear(self.feature_extractor.output_feature_len, 2) for _ in range(domain_num)])

        # Gradient reversal layer.
        self.grls = [GradientReversalLayer() for _ in range(domain_num)]

        self._initialize_weights()

    def forward(self, source_inputs_list, target_inputs):
        """
        :param sinputs:     A list of k inputs from k source domains.
        :param tinputs:     Input from the target domain.
        :return:
        """
        ## ===================================
        # Extract features from each domain
        ## ===================================
        source_num = len(source_inputs_list)
        source_feature_list = []
        for source_inputs in source_inputs_list:
            #print(source_inputs.size())
            feature = self.feature_extractor(source_inputs)
            source_feature_list.append(feature)
        
        target_feature = self.feature_extractor(target_inputs)

        # Classification probabilities on k source domains.
        logprobs = []
        for source_feature in source_feature_list:
            logprobs.append(F.log_softmax(self.classifier(source_feature), dim=1))

        # Domain classification accuracies.
        source_pred, target_pred = [], []
        for i in range(source_num):
            source_pred.append(F.log_softmax(self.domain_classifier[i](self.grls[i](source_feature_list[i])), dim=1))
            target_pred.append(F.log_softmax(self.domain_classifier[i](self.grls[i](target_feature)), dim=1))
        return logprobs, source_pred, target_pred

    def inference(self, inputs):
        features = self.feature_extractor(inputs)
        # Classification probability.
        logprobs = F.log_softmax(self.classifier(features), dim=1)
        return logprobs
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.zero_()

extractor_mapping = {
    'alexnet': alexnet_extractor(pretrained=True),
    'vgg16_bn': vgg16_extractor(pretrained=True),
    'resnet50': resnet50_extractor(pretrained=True)
}
def load_model(name, class_num, domain_num, extractor = 'alexnet'):
    feature_extractor = extractor_mapping[extractor]
    if name == 'mdan':
        return MDANet(class_num, domain_num, feature_extractor)


def test():
    extractor = 'resnet50'
    img = torch.rand(1,3,224,224)
    x = [img, img, img]
    model = load_model('mdan', 345, 3, extractor = extractor)
    logprobs, source_pred, target_pred = model(x, img)
    print(logprobs.size(), source_pred.size(), target_pred.size())
    