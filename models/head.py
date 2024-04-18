
import torch
import torch.nn as nn
import utils
import torch.nn.functional as F

from utils import trunc_normal_

class CSyncBatchNorm(nn.SyncBatchNorm):
    def __init__(self,
                 *args,
                 with_var=False,
                 **kwargs):
        super(CSyncBatchNorm, self).__init__(*args, **kwargs)
        self.with_var = with_var

    def forward(self, x):
        # center norm
        self.training = False
        if not self.with_var:
            self.running_var = torch.ones_like(self.running_var)
        normed_x = super(CSyncBatchNorm, self).forward(x)
        # udpate center
        self.training = True
        _ = super(CSyncBatchNorm, self).forward(x)
        return normed_x

class PSyncBatchNorm(nn.SyncBatchNorm):
    def __init__(self,
                 *args,
                 bunch_size,
                 **kwargs):
        procs_per_bunch = min(bunch_size, utils.get_world_size())
        assert utils.get_world_size() % procs_per_bunch == 0
        n_bunch = utils.get_world_size() // procs_per_bunch
        #
        ranks = list(range(utils.get_world_size()))
        print('---ALL RANKS----\n{}'.format(ranks))
        rank_groups = [ranks[i*procs_per_bunch: (i+1)*procs_per_bunch] for i in range(n_bunch)]
        print('---RANK GROUPS----\n{}'.format(rank_groups))
        process_groups = [torch.distributed.new_group(pids) for pids in rank_groups]
        bunch_id = utils.get_rank() // procs_per_bunch
        process_group = process_groups[bunch_id]
        print('---CURRENT GROUP----\n{}'.format(process_group))
        super(PSyncBatchNorm, self).__init__(*args, process_group=process_group, **kwargs)

class CustomSequential(nn.Sequential):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    def forward(self, input):
        for module in self:
            dim = len(input.shape)
            if isinstance(module, self.bn_types) and dim > 2:
                perm = list(range(dim - 1)); perm.insert(1, dim - 1)
                inv_perm = list(range(dim)) + [1]; inv_perm.pop(1)
                input = module(input.permute(*perm)).permute(*inv_perm)
            else:
                input = module(input)
        return input

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, norm=None, act='gelu', last_norm=None, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True, **kwargs):
        super().__init__()
        norm = self._build_norm(norm, hidden_dim)
        last_norm = self._build_norm(last_norm, out_dim, affine=False, **kwargs)
        act = self._build_act(act)

        nlayers = max(nlayers, 1)
        if nlayers == 1:
            if bottleneck_dim > 0:
                self.mlp = nn.Linear(in_dim, bottleneck_dim)
            else:
                self.mlp = nn.Linear(in_dim, out_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if norm is not None:
                layers.append(norm)
            layers.append(act)
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if norm is not None:
                    layers.append(norm)
                layers.append(act)
            if bottleneck_dim > 0:
                layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            else:
                layers.append(nn.Linear(hidden_dim, out_dim))
            self.mlp = CustomSequential(*layers)
        self.apply(self._init_weights)
        
        if bottleneck_dim > 0:
            self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False
        else:
            self.last_layer = None

        self.last_norm = last_norm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        if self.last_layer is not None:
            x = nn.functional.normalize(x, dim=-1, p=2)
            x = self.last_layer(x)
        if self.last_norm is not None:
            x = self.last_norm(x)
        return x

    def _build_norm(self, norm, hidden_dim, **kwargs):
        if norm == 'bn':
            norm = nn.BatchNorm1d(hidden_dim, **kwargs)
        elif norm == 'syncbn':
            norm = nn.SyncBatchNorm(hidden_dim, **kwargs)
        elif norm == 'csyncbn':
            norm = CSyncBatchNorm(hidden_dim, **kwargs)
        elif norm == 'psyncbn':
            norm =  PSyncBatchNorm(hidden_dim, **kwargs)
        elif norm == 'ln':
            norm = nn.LayerNorm(hidden_dim, **kwargs)
        else:
            assert norm is None, "unknown norm type {}".format(norm)
        return norm

    def _build_act(self, act):
        if act == 'relu':
            act = nn.ReLU()
        elif act == 'gelu':
            act = nn.GELU()
        else:
            assert False, "unknown act type {}".format(act)
        return act

class iBOTHead(DINOHead):

    def __init__(self, *args, patch_out_dim=8192, norm=None, act='gelu', last_norm=None, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True, 
                 shared_head=False, **kwargs):
        
        super(iBOTHead, self).__init__(*args,
                                        norm=norm,
                                        act=act,
                                        last_norm=last_norm,
                                        nlayers=nlayers,
                                        hidden_dim=hidden_dim,
                                        bottleneck_dim=bottleneck_dim,
                                        norm_last_layer=norm_last_layer, 
                                        **kwargs)

        if not shared_head:
            if bottleneck_dim > 0:
                self.last_layer2 = nn.utils.weight_norm(nn.Linear(bottleneck_dim, patch_out_dim, bias=False))
                self.last_layer2.weight_g.data.fill_(1)
                if norm_last_layer:
                    self.last_layer2.weight_g.requires_grad = False
            else:
                self.mlp2 = nn.Linear(hidden_dim, patch_out_dim)
                self.last_layer2 = None

            self.last_norm2 = self._build_norm(last_norm, patch_out_dim, affine=False, **kwargs)
        else:
            if bottleneck_dim > 0:
                self.last_layer2 = self.last_layer
            else:
                self.mlp2 = self.mlp[-1]
                self.last_layer2 = None

            self.last_norm2 = self.last_norm

    def forward(self, x):
        if len(x.shape) == 2:
            return super(iBOTHead, self).forward(x)

        if self.last_layer is not None:
            x = self.mlp(x) # return [2B, 197, 384] -> [2B, 197, 256]
            x = nn.functional.normalize(x, dim=-1, p=2)
            x1 = self.last_layer(x[:, 0]) # cls_token [2B, 8192]
            x2 = self.last_layer2(x[:, 1:]) # patch_token [2B, 196, 8192]
        else:
            x = self.mlp[:-1](x)
            x1 = self.mlp[-1](x[:, 0])
            x2 = self.mlp2(x[:, 1:])
        
        if self.last_norm is not None:
            x1 = self.last_norm(x1)
            x2 = self.last_norm2(x2)
        
        return x1, x2


class linSeg(nn.Module):
    # linear seg from FCN head
    def __init__(self, embed_dim, num_classes, img_dim=224, patch_dim=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear')
        self.name = 'linseg'

        self.conv_seg = nn.Conv2d(self.embed_dim*4, num_classes, kernel_size=1)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embed_dim,
        ) # [B, 196, C] -> [B, 14, 14, C]
        x = x.permute(0, 3, 1, 2).contiguous() # [B, 14, 14, C] -> [B, C, 14, 14]
        return x

    def forward(self, feats:list, *args):
        inputs = [self._reshape_output(item) for item in feats]
        inputs = torch.cat(inputs, dim=1) # [B, 4C, H, W]
        output = self.conv_seg(inputs)
        output = self.upsample(output) # x16
        return output

class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs


class ClsHead(nn.Module):
    '''
    Wu, Jianfang, et al. "Vision Transformer‐based recognition of diabetic retinopathy grade." Medical Physics 48.12 (2021): 7850-7863.
    '''
    def __init__(self, embed_dim, num_classes, layers=3):
        super(ClsHead, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.layers = layers # default is 3 layers, we test different layers for retfound

        if self.layers == 3:
            channels = [self.embed_dim, self.embed_dim//2, self.embed_dim//4, self.num_classes]
            self.classifier = nn.Sequential(
                nn.Linear(channels[0], channels[1]),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(channels[1], channels[2]),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(channels[2], channels[3])
            )
        elif self.layers == 2:
            channels = [self.embed_dim, self.embed_dim//4, self.num_classes]
            self.classifier = nn.Sequential(
                nn.Linear(channels[0], channels[1]),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(channels[1], channels[2])
            )
        elif self.layers == 1:
            channels = [self.embed_dim, self.num_classes]
            self.classifier = nn.Sequential(
                nn.Linear(channels[0], channels[1]),
            )
        self.channel_bn = nn.BatchNorm2d(
            self.embed_dim,
            eps=1e-6, # default 1e-6
            momentum=0.99, # default: 0.99
        )
        self.init_weights()

    def init_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias.data, 0.0)
                nn.init.normal_(m.weight.data, mean=0.0, std=0.01)


    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(2).unsqueeze(3)
        # flatten
        x = self.channel_bn(x)
        x = x.view(x.size(0), -1)
        # linear layer
        return self.classifier(x)

class RegHead(nn.Module):
    '''
    regression head
    '''
    def __init__(self, embed_dim, num_classes):
        super(RegHead, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        channels = [self.embed_dim, self.embed_dim//2, self.embed_dim//4, self.num_classes]

        self.classifier = nn.Sequential(
            nn.Linear(channels[0], channels[2]),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(channels[2], channels[3]),
        )
        self.channel_bn = nn.BatchNorm2d(
            self.embed_dim,
            eps=1e-6, # default 1e-6
            momentum=0.99, # default: 0.99
        )
        self.init_weights()

    def init_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias.data, 0.0)
                nn.init.normal_(m.weight.data, mean=0.0, std=0.01)


    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(2).unsqueeze(3)
        # flatten
        x = self.channel_bn(x)
        x = x.view(x.size(0), -1)
        # linear layer
        return self.classifier(x)


class ForecastHead(nn.Module):
    def __init__(self, input_dim, max_len):
        super(ForecastHead, self).__init__()
        self.input_dim = input_dim
        self.fc_1 = nn.Linear(self.input_dim, 1)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dim, 2) * (-math.log(10000.0) / input_dim))
        pe = torch.zeros(max_len, 1, input_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, times):
        x = torch.add(x, torch.squeeze(self.pe[times]))
        return self.fc_1(x)
