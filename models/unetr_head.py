# the UnetR head for the downstream segmentation task
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.deconv(x)

class Unetr_Head(nn.Module):
    # default for vit-base model
    def __init__(self,embed_dim, num_classes, img_dim=224, patch_dim=16, dropout_rate=0.3):
        super(Unetr_Head, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.dropout_rate = dropout_rate
        self.patch_dim = patch_dim
        self.name = 'unetr'

        self.base_channel = 32
        print(f"base channel: {self.base_channel}")

        # add at 09/15
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout2d(self.dropout_rate)
        else:
            self.dropout = None

        """ CNN Decoder """
        ## Decoder 1
        self.d1 = DeconvBlock(self.embed_dim, self.base_channel*4)
        self.s1 = nn.Sequential(
            DeconvBlock(self.embed_dim, self.base_channel*4),
            ConvBlock(self.base_channel*4, self.base_channel*4)
        )
        self.c1 = nn.Sequential(
            ConvBlock(self.base_channel*8, self.base_channel*4),
            ConvBlock(self.base_channel*4, self.base_channel*4)
        )

        ## Decoder 2
        self.d2 = DeconvBlock(self.base_channel*4, self.base_channel*2)
        self.s2 = nn.Sequential(
            DeconvBlock(self.embed_dim, self.base_channel*2),
            ConvBlock(self.base_channel*2, self.base_channel*2),
            DeconvBlock(self.base_channel*2, self.base_channel*2),
            ConvBlock(self.base_channel*2, self.base_channel*2)
        )
        self.c2 = nn.Sequential(
            ConvBlock(self.base_channel*4, self.base_channel*2),
            ConvBlock(self.base_channel*2, self.base_channel*2)
        )

        ## Decoder 3
        self.d3 = DeconvBlock(self.base_channel*2, self.base_channel)
        self.s3 = nn.Sequential(
            DeconvBlock(self.embed_dim, self.base_channel),
            ConvBlock(self.base_channel, self.base_channel),
            DeconvBlock(self.base_channel, self.base_channel),
            ConvBlock(self.base_channel, self.base_channel),
            DeconvBlock(self.base_channel, self.base_channel),
            ConvBlock(self.base_channel, self.base_channel)
        )
        self.c3 = nn.Sequential(
            ConvBlock(self.base_channel*2, self.base_channel),
            ConvBlock(self.base_channel, self.base_channel)
        )

        ## Decoder 4
        self.d4 = DeconvBlock(self.base_channel, self.base_channel//2)
        self.s4 = nn.Sequential(
            ConvBlock(3, self.base_channel//2),
            ConvBlock(self.base_channel//2, self.base_channel//2)
        )
        self.c4 = nn.Sequential(
            ConvBlock(self.base_channel, self.base_channel//2),
            ConvBlock(self.base_channel//2, self.base_channel//2)
        )

        """ Output """
        self.cls = nn.Conv2d(self.base_channel//2, self.num_classes, kernel_size=1, padding=0)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embed_dim,
        ) # [B, 196, C] -> [B, 14, 14, C]
        x = x.permute(0, 3, 1, 2).contiguous() # [B, 14, 14, C] -> [B, C, 14, 14]
        return x

    def forward(self, feats:list, inputs):
        z3, z6, z9, z12 = feats

        ## Reshaping
        batch = inputs.shape[0]
        z0 = inputs # [B, C, H, W]
        z3, z6, z9, z12 = self._reshape_output(z3), self._reshape_output(z6),self._reshape_output(z9),self._reshape_output(z12),

        ## Decoder 1
        x = self.d1(z12) # x2, deconv
        s = self.s1(z9) # deconv + conv
        x = torch.cat([x, s], dim=1)
        x = self.c1(x) # conv

        ## Decoder 2
        x = self.d2(x)
        s = self.s2(z6)
        x = torch.cat([x, s], dim=1)
        x = self.c2(x)

        ## Decoder 3
        x = self.d3(x)
        s = self.s3(z3)
        x = torch.cat([x, s], dim=1)
        x = self.c3(x)

        ## Decoder 4
        x = self.d4(x)
        s = self.s4(z0)
        x = torch.cat([x, s], dim=1)
        x = self.c4(x)

        """ Output """
        if self.dropout is not None:
            x = self.dropout(x) # add dropout
        output = self.cls(x) # [B, C, H, W]

        return output
