import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import copy

from ViT import ViT


class VEATIC_baseline(nn.Module):
    def __init__(self, 
                 num_frames = 5,
                 num_classes = 2,
                 dim = 2048,
                 depth = 6,
                 heads = 16,
                 mlp_dim = 2048,
                 dropout = 0.4, 
                 backnone="resnet50"):
        super(VEATIC_baseline, self).__init__()
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        if backnone == "resnet50":
            pretrained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            pretrained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.feature = nn.Sequential(pretrained_model.conv1, pretrained_model.bn1,pretrained_model.relu,
            pretrained_model.maxpool,pretrained_model.layer1,pretrained_model.layer2,pretrained_model.layer3)
        
        self.human = copy.deepcopy(pretrained_model.layer4)
        
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.ViT = ViT(
                    num_frames = self.num_frames,
                    num_classes = self.num_classes,
                    dim = self.dim,
                    depth = self.depth,
                    heads = self.heads,
                    mlp_dim = self.mlp_dim,
                    dropout = self.dropout,
                )
    
    def forward(self, frames):
        '''
            one stream +vit
        '''
        frame_features = []
        N = frames.shape[1]
        for i in range(N):
            x = self.feature(frames[:, i, :, :, :])
            x_human = self.human(x)
            x_out = self.pool(x_human)
            x_out = x_out.reshape(frames.shape[0], -1)
            frame_features.append(x_out)

        out = self.ViT(torch.stack(frame_features).permute(1, 0, 2))

        return out


if __name__ == '__main__':
    model = VEATIC_baseline()
    frames = torch.randn((2,5,3,640,480))
    out = model(frames)
    
    print(out.shape)