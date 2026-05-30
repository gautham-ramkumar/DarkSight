import torch
import torch.nn as nn
import torch.nn.functional as F

class SuperPoint(nn.Module):
    """ 
    SuperPoint: Self-Supervised Interest Point Detection and Description
    Paper: https://arxiv.org/abs/1712.07629
    
    This implementation matches the official pretrained model architecture
    where keys are like 'conv1a.weight'.
    """
    def __init__(self):
        super(SuperPoint, self).__init__()
        
        # Shared Encoder
        self.conv1a = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # Detector Head
        self.convPa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)

        # Descriptor Head
        self.convDa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """ 
        x: (B, 1, H, W) grayscale in [0, 1]
        Returns:
            semi: [B, 65, H/8, W/8]
            desc: [B, 256, H/8, W/8]
        """
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))

        # Detector Head
        cPa = F.relu(self.convPa(x))
        semi = self.convPb(cPa)

        # Descriptor Head
        cDa = F.relu(self.convDa(x))
        desc = self.convDb(cDa)
        desc = F.normalize(desc, p=2, dim=1)

        return semi, desc
