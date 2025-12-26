import torch.nn as nn
import torch
from .utils_IWO import (
    MBRConv5,
    MBRConv3,
    MBRConv1,
    DropBlock
)
from .utils_IWO_v2 import (
    RetinexFST, 
    RetinexFSTS, 
    RetinexHDPA,
    RetinexHDPAS
)

class MobileRetinexV2ISPNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, channels=32, rep_scale=4):
        super(MobileRetinexV2ISPNet, self).__init__()
        self.channels = channels

        # This gives us Receptive Field (Conv5) AND Quadratic Curve (FST) immediately.
        self.head_conv = MBRConv5(in_channels, channels, rep_scale=rep_scale)
        self.head_fst = RetinexFST(channels, rep_scale=rep_scale)
        
        # Body (Retinex Enhanced)
        self.body1 = nn.Sequential(
            MBRConv3(channels, channels, rep_scale=rep_scale),
            RetinexFST(channels, rep_scale=rep_scale),
            RetinexHDPA(channels, rep_scale=rep_scale)
        )
        self.body2 = nn.Sequential(
            MBRConv3(channels, channels, rep_scale=rep_scale),
            RetinexFST(channels, rep_scale=rep_scale),
            RetinexHDPA(channels, rep_scale=rep_scale)
        )
        
        # Tail
        self.tail = nn.Sequential(
            MBRConv1(channels, channels, rep_scale=rep_scale),
            nn.PixelShuffle(2),
            MBRConv1(channels // 4, out_channels, rep_scale=rep_scale),
            nn.Sigmoid() 
        )
        self.tail_warm = MBRConv3(channels, 4, rep_scale=rep_scale)
        self.drop = DropBlock(3)

    def forward(self, x):
        x_feat = self.head_conv(x)
        x_shallow = self.head_fst(x_feat)

        x1 = self.body1(x_shallow)
        x1 = x1 + x_shallow
        
        x2 = self.body2(x1)
        x2 = x2 + x1
        
        return self.tail(x2)

    def forward_warm(self, x):
        x = self.drop(x)
        x = self.head_conv(x)
        x = self.head_fst(x)
        x = self.body1(x)
        x = self.body2(x)
        return self.tail(x), self.tail_warm(x)

    def slim(self):
        net_slim = MobileRetinexV2ISPNetS(channels=self.channels)
        weight_slim = net_slim.state_dict()
        for name, mod in self.named_modules():
            if isinstance(mod, MBRConv3) or isinstance(mod, MBRConv5) or isinstance(mod, MBRConv1):
                if '%s.weight' % name in weight_slim:
                    w, b = mod.slim()
                    weight_slim['%s.weight' % name] = w 
                    weight_slim['%s.bias' % name] = b 
            elif isinstance(mod, RetinexFST):
                weight_slim['%s.bias' % name] = mod.bias
            elif isinstance(mod, nn.PReLU):
                weight_slim['%s.weight' % name] = mod.weight
        net_slim.load_state_dict(weight_slim)
        return net_slim

class MobileRetinexV2ISPNetS(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, channels=32):
        super(MobileRetinexV2ISPNetS, self).__init__()
        
        # Head: Conv5 + RetinexFST
        self.head_conv = nn.Conv2d(in_channels, channels, 5, 1, 2)
        self.head_fst = RetinexFSTS(channels)
        
        self.body1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            RetinexFSTS(channels),
            RetinexHDPAS(channels)
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            RetinexFSTS(channels),
            RetinexHDPAS(channels)
        )
        
        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        x_feat = self.head_conv(x)
        x_shallow = self.head_fst(x_feat)
        
        x1 = self.body1(x_shallow)
        x1 = x1 + x_shallow
        
        x2 = self.body2(x1)
        x2 = x2 + x1
        
        return self.tail(x2)
