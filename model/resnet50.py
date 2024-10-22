import torch
import torch.nn as nn
import torchvision.models as models

class UNetResNet50(nn.Module):
    def __init__(self, num_classes):
        super(UNetResNet50, self).__init__()
        
        # Load the ResNet50 model
        self.resnet = models.resnet50(pretrained=True)
        
        # Encoder
        self.encoder1 = nn.Sequential(*list(self.resnet.children())[:3])
        self.encoder2 = nn.Sequential(*list(self.resnet.children())[3:5])
        self.encoder3 = self.resnet.layer1
        self.encoder4 = self.resnet.layer2
        self.encoder5 = self.resnet.layer3
        self.encoder6 = self.resnet.layer4
        
        # Decoder
        self.upconv6 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.decoder6 = self._block(2048, 1024)
        
        self.upconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder5 = self._block(1024, 512)
        
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = self._block(512, 256)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = self._block(256, 128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = self._block(128, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = self._block(67, 64)
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def _block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        enc6 = self.encoder6(enc5)
        
        # Decoder
        dec6 = self.upconv6(enc6)
        dec6 = torch.cat((dec6, enc5), dim=1)
        dec6 = self.decoder6(dec6)
        
        dec5 = self.upconv5(dec6)
        dec5 = torch.cat((dec5, enc4), dim=1)
        dec5 = self.decoder5(dec5)
        
        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc3), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(dec1)
        
        return self.final_conv(dec1)
