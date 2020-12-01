import torch
import torch.nn as nn
from torch.autograd import Function

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd, reverse=True):
        ctx.lambd = lambd
        ctx.reverse=reverse
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.reverse:
            return (grad_output * -ctx.lambd), None, None
        else:
            return (grad_output * ctx.lambd), None, None

def grad_reverse(x, lambd=1.0, reverse=True):
    return GradReverse.apply(x, lambd, reverse)

class DisNet(nn.Module):
    def __init__(self, in_channels, num_domains, layers=[1024, 256]):
        super(DisNet, self).__init__()
        self.domain_classifier = nn.ModuleList()
        
        self.domain_classifier.append(nn.Linear(in_channels, layers[0]))
        for i in range(1, len(layers)):    
            self.domain_classifier.append(nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Linear(layers[i-1], layers[i])))
        self.domain_classifier.append(nn.ReLU(inplace=True))
        self.domain_classifier.append(nn.Dropout())
        self.domain_classifier.append(nn.Linear(layers[-1], num_domains))
        self.domain_classifier = nn.Sequential(*self.domain_classifier)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, .1)
                nn.init.constant_(m.bias, 0.)

        self.lambda_ = 0.0

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

    def forward(self, x):
        x = grad_reverse(x, self.lambda_)
        return self.domain_classifier(x)

    def get_params(self, lr):
        return [{"params": self.domain_classifier.parameters(), "lr": lr}]

class ClsNet(nn.Module):
    def __init__(self, in_channels, num_domains, num_classes, reverse=True, layers=[1024, 256]):
        super(ClsNet, self).__init__()
        self.classifier_list = nn.ModuleList()
        for _ in range(num_domains):
            class_list = nn.ModuleList()
            class_list.append(nn.Linear(in_channels, layers[0]))
            for i in range(1, len(layers)):
                class_list.append(nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Linear(layers[i-1], layers[i])
                ))
            class_list.append(nn.ReLU(inplace=True))
            class_list.append(nn.Dropout())
            class_list.append(nn.Linear(layers[-1], num_classes))
            self.classifier_list.append(nn.Sequential(*class_list))
        for m in self.classifier_list.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, .1)
                nn.init.constant_(m.bias, 0.)

        self.lambda_ = 0
        self.reverse = reverse
 
    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

    def forward(self, x):
        output = []
        for c, x_ in zip(self.classifier_list, x):
            if len(x_) == 0:
                output.append(None)
            else:
                x_ = grad_reverse(x_, self.lambda_, self.reverse)
                output.append(c(x_))

        return output

    def get_params(self, lr):
        return [{"params": self.classifier_list.parameters(), "lr": lr}]

def aux_models(in_channels, num_domains, num_classes, layers_dis=[], layers_cls=[]):
    
    dis_model = DisNet(in_channels, num_domains, layers_dis)
    c_model = ClsNet(in_channels, num_domains, num_classes, reverse=False, layers=layers_cls)
    cp_model = ClsNet(in_channels, num_domains, num_classes, reverse=True, layers=layers_cls)

    return dis_model, c_model, cp_model
