import torch
import torch.nn as nn

class FlatZero3DCE(nn.Module):
    """
    Original 7-layer flat 3D convolutional network.
    Operates at full resolution throughout (no pooling/upsampling).
    Baseline architecture for v1.
    """
    def __init__(self, n_iter: int = 8):
        super().__init__()
        self.n_iter = n_iter
        # Standard 3D convolutions with padding to maintain spatial/temporal dims
        self.conv1 = nn.Sequential(nn.Conv3d(3,  32, 3, padding=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(32, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv3d(32, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv3d(32, 32, 3, padding=1), nn.ReLU(inplace=True))
        
        # Skip connections via concatenation (32+32=64 channels)
        self.conv5 = nn.Sequential(nn.Conv3d(64, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv3d(64, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.conv7 = nn.Conv3d(64, 3 * n_iter, 3, padding=1)

    def forward(self, x):
        # x: (B, 3, D, H, W)
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        
        # Decoder-style skips in a flat architecture
        f5 = self.conv5(torch.cat([f3, f4], dim=1))
        f6 = self.conv6(torch.cat([f2, f5], dim=1))
        
        # Produce alpha maps (A) and apply LEQ enhancement curve
        A = torch.tanh(self.conv7(torch.cat([f1, f6], dim=1)))
        
        out = x
        for i in range(self.n_iter):
            out = out + A[:, i*3:(i+1)*3] * (out - out**2)
            
        return A, out.clamp(0.0, 1.0)
