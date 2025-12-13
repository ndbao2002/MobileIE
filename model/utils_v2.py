import torch
import torch.nn as nn
from torch.nn import functional as F
from .utils import MBRConv3, MBRConv5, MBRConv1

###################################################################################
class RetinexFST(nn.Module):
    def __init__(self, channels, rep_scale=4):
        super(RetinexFST, self).__init__()
        
        # 1. Main Feature Extraction (Reflectance/Detail Candidate)
        # This corresponds to the 'block1' in original FST
        self.conv_main = MBRConv3(channels, channels, rep_scale=rep_scale)
        
        # 2. Illumination Estimator (Global Context)
        # Replaces the expensive second convolution with a cheap Global Gating
        self.illumination_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            MBRConv1(channels, channels, rep_scale=rep_scale), # 1x1 Conv is cheap
            nn.Sigmoid()
        )
        
        # 3. The "Small Bias Trick" (MAI 2025)
        # Acts as Ambient Light / DC Offset
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # 1. Extract Spatially Rich Features (Reflectance proxy)
        x_feat = self.conv_main(x)
        
        # 2. Estimate Global Illumination (L)
        # "How bright is this channel globally?"
        illum = self.illumination_gate(x_feat)
        
        # 3. Retinex Fusion: I = L * R + Bias
        # We modulate the features by the illumination gate
        out = x_feat * illum + self.bias
        
        # Residual connection is handled in the main Net, but FST usually transforms 'x'
        # Original FST was: w1*x1 * w2*x1 + bias. 
        # Ours is: x1 * Global(x1) + bias. 
        # This is physically cleaner and computationally lighter.
        return out
    
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
        
        # Global Illumination Path
        self.global_pool = nn.AdaptiveAvgPool2d(16) # Maintain vignette info
        self.global_fc = nn.Sequential(
            MBRConv1(channels, channels // 2, rep_scale=rep_scale),
            nn.PReLU(),
            MBRConv1(channels // 2, channels, rep_scale=rep_scale),
            nn.Sigmoid()
        )
        
        # Local Reflectance Path
        self.local_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.local_fc = nn.Sequential(
            MBRConv3(channels, channels // 2, rep_scale=rep_scale),
            nn.PReLU(),
            MBRConv3(channels // 2, channels, rep_scale=rep_scale),
            nn.Sigmoid()
        )

        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        g = F.interpolate(self.global_fc(self.global_pool(x)), size=x.shape[2:], mode='bilinear')
        l = self.local_fc(self.local_pool(x))
        return x * g * l + self.bias
    
class RetinexHDPAS(nn.Module):
    def __init__(self, channels):
        super(RetinexHDPAS, self).__init__()
        
        # Global Illumination Path
        self.global_pool = nn.AdaptiveAvgPool2d(16) # Maintain vignette info
        self.global_fc = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.PReLU(),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )
        
        # Local Reflectance Path
        self.local_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.local_fc = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(channels // 2, channels, 3, 1, 1),
            nn.Sigmoid()
        )

        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        g = F.interpolate(self.global_fc(self.global_pool(x)), size=x.shape[2:], mode='bilinear')
        l = self.local_fc(self.local_pool(x))
        return x * g * l + self.bias