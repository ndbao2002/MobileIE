import torch
import torch.nn as nn
from torch.nn import functional as F
from .utils_IWO import MBRConv3, MBRConv5, MBRConv1

###################################################################################
class RetinexFST(nn.Module):
    def __init__(self, channels, rep_scale=4):
        super(RetinexFST, self).__init__()
        # Main feature extraction (Reflectance/Detail Candidate)
        self.conv_main = MBRConv3(channels, channels, rep_scale=rep_scale)
        
        # Illumination Estimator (Global Context)
        # Replaces expensive second Conv with cheap AvgPool + 1x1 Conv
        self.illumination_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            MBRConv1(channels, channels, rep_scale=rep_scale), 
            nn.Sigmoid()
        )
        
        # The "Small Bias Trick" (MAI 2025) - Additive Ambient Light
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        x_feat = self.conv_main(x)
        illum = self.illumination_gate(x_feat)
        # Retinex Physics: Multiply by Light, Add Ambient Bias
        return x_feat * illum + self.bias

class RetinexFSTS(nn.Module):
    def __init__(self, channels):
        super(RetinexFSTS, self).__init__()
        self.conv_main = nn.Conv2d(channels, channels, 3, 1, 1)
        self.illumination_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        x_feat = self.conv_main(x)
        illum = self.illumination_gate(x_feat)
        return x_feat * illum + self.bias
    
###################################################################################
class RetinexHDPA(nn.Module):
    def __init__(self, channels, rep_scale=4):
        super(RetinexHDPA, self).__init__()
        
        # Global Path (Illumination/Vignette) - Keeps spatial grid
        self.global_pool = nn.AdaptiveAvgPool2d(16) 
        self.global_fc = nn.Sequential(
            MBRConv1(channels, channels, rep_scale=rep_scale), 
            nn.Sigmoid()
        )
        
        # Local Path (Reflectance/Texture)
        # Optimized: Single 1x1 Conv after MaxPool (Fastest possible edge attention)
        self.local_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.local_fc = nn.Sequential(
            MBRConv1(channels, channels, rep_scale=rep_scale), 
            nn.Sigmoid()
        )

    def forward(self, x):
        # Global Map (Upsample)
        g_small = self.global_fc(self.global_pool(x))
        g = F.interpolate(g_small, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Local Map
        l = self.local_fc(self.local_pool(x))
        
        # Fusion
        return x * g * l
    
class RetinexHDPAS(nn.Module):
    def __init__(self, channels):
        super(RetinexHDPAS, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(16)
        self.global_fc = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        self.local_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.local_fc = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        g = F.interpolate(self.global_fc(self.global_pool(x)), size=x.shape[2:], mode='bilinear', align_corners=False)
        l = self.local_fc(self.local_pool(x))
        return x * g * l