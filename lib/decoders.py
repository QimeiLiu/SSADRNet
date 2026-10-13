import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        #print(x.shape)
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size=3, stride=1, padding=1, groups=1):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=kernel_size,stride=stride,padding=padding,bias=True),
		    nn.BatchNorm2d(ch_out),
	            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

        
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class SkipSelectiveFusion(nn.Module):
    """AG-preserving skip fusion with a zero-init selective residual branch."""

    def __init__(self, F_g, F_l, F_int, ratio=8):
        super(SkipSelectiveFusion, self).__init__()
        self.AG = Attention_block(F_g=F_g, F_l=F_l, F_int=F_int)
        hidden = max(F_l // ratio, 16)
        self.route = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(F_l * 2, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, F_l * 2, kernel_size=1, bias=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(F_l, F_l, kernel_size=3, stride=1, padding=1, groups=F_l, bias=False),
            nn.BatchNorm2d(F_l),
            nn.GELU(),
            nn.Conv2d(F_l, F_l, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_l),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, g, x):
        x_ag = self.AG(g=g, x=x)
        base = g + x_ag
        b, c = g.size(0), g.size(1)
        route_in = torch.cat((g, x_ag), dim=1)
        w = self.route(route_in).view(b, 2, c, 1, 1)
        w = torch.softmax(w, dim=1)
        mixed = w[:, 0] * g + w[:, 1] * x_ag
        return base + self.gate * self.refine(mixed)


class SkipResidualAdapter(nn.Module):
    """Zero-init side adapter after the original AG skip addition."""

    def __init__(self, ch, ratio=8):
        super(SkipResidualAdapter, self).__init__()
        hidden = max(ch // ratio, 16)
        self.route = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch * 2, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, ch * 2, kernel_size=1, bias=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, groups=ch, bias=False),
            nn.BatchNorm2d(ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, agg, skip_ag):
        b, c = agg.size(0), agg.size(1)
        w = self.route(torch.cat((agg, skip_ag), dim=1)).view(b, 2, c, 1, 1)
        w = torch.softmax(w, dim=1)
        mixed = w[:, 0] * agg + w[:, 1] * skip_ag
        return agg + self.gate * self.refine(mixed)


class BoundedSkipResidualAdapter(nn.Module):
    """Conservative skip adapter with a bounded residual scale."""

    def __init__(self, ch, ratio=8, max_scale=0.5):
        super(BoundedSkipResidualAdapter, self).__init__()
        hidden = max(ch // ratio, 16)
        self.route = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch * 2, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, ch * 2, kernel_size=1, bias=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, groups=ch, bias=False),
            nn.BatchNorm2d(ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.gate = nn.Parameter(torch.zeros(1))
        self.max_scale = max_scale

    def forward(self, agg, skip_ag):
        b, c = agg.size(0), agg.size(1)
        w = self.route(torch.cat((agg, skip_ag), dim=1)).view(b, 2, c, 1, 1)
        w = torch.softmax(w, dim=1)
        mixed = w[:, 0] * agg + w[:, 1] * skip_ag
        scale = self.max_scale * torch.tanh(self.gate)
        return agg + scale * self.refine(mixed)


class JointChannelRecalibration(nn.Module):
    """Joint decoder/skip channel gate initialized as an exact identity."""

    def __init__(self, ch, ratio=8, max_delta=0.5):
        super(JointChannelRecalibration, self).__init__()
        hidden = max(ch // ratio, 16)
        self.reduce = nn.Conv2d(ch * 2, hidden, kernel_size=1, bias=True)
        self.expand = nn.Conv2d(hidden, ch, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.max_delta = max_delta
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, decoder, skip_ag):
        context = torch.cat((decoder, skip_ag), dim=1)
        context = F.adaptive_avg_pool2d(context, 1)
        delta = self.max_delta * torch.tanh(self.expand(self.act(self.reduce(context))))
        return decoder + skip_ag * (1.0 + delta)


class ComplementaryChannelFusion(nn.Module):
    """Boundary-aware complementary fusion initialized as the exact AB sum.

    Channel context controls which semantic features need rebalancing, while the
    spatial branch localizes that correction to decoder/skip disagreement regions.
    Both prediction heads are zero initialized, so adding C does not perturb the
    original ``decoder + attended_skip`` path at initialization.
    """

    def __init__(self, ch, ratio=8, max_delta=0.25):
        super(ComplementaryChannelFusion, self).__init__()
        hidden = max(ch // ratio, 16)
        self.reduce = nn.Conv2d(ch * 2, hidden, kernel_size=1, bias=True)
        self.expand = nn.Conv2d(hidden, ch, kernel_size=1, bias=True)
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
        self.act = nn.GELU()
        self.max_delta = max_delta
        self.channel_only = False
        self.spatial_only = False
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)
        nn.init.zeros_(self.spatial.weight)
        nn.init.zeros_(self.spatial.bias)

    def forward(self, decoder, skip_ag):
        context = torch.cat((decoder, skip_ag), dim=1)
        context = F.adaptive_avg_pool2d(context, 1)
        channel_delta = torch.tanh(self.expand(self.act(self.reduce(context))))

        disagreement = torch.abs(skip_ag - decoder)
        spatial_context = torch.cat(
            (disagreement.mean(dim=1, keepdim=True),
             disagreement.amax(dim=1, keepdim=True)), dim=1)
        spatial_delta = torch.tanh(self.spatial(spatial_context))

        # The sum is bounded to [-max_delta, max_delta] and can independently
        # select semantic channels and uncertain boundary locations.
        if self.channel_only:
            delta = self.max_delta * channel_delta
        elif self.spatial_only:
            delta = self.max_delta * spatial_delta
        else:
            delta = 0.5 * self.max_delta * (channel_delta + spatial_delta)
        return decoder + skip_ag + delta * (skip_ag - decoder)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool_out)))
        #print(x.shape)
        max_pool_out= x.amax(dim=(2, 3), keepdim=True) # was AdaptiveMaxPool2d(1); amax backward is deterministic

        max_out = self.fc2(self.relu1(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out) 

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)    

class CASCADE_Cat(nn.Module):
    def __init__(self, channels=[512,320,128,64]):
        super(CASCADE_Cat,self).__init__()
        
        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])
	
        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=2*channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=2*channels[2], ch_out=channels[2])
        
        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3],F_l=channels[3],F_int=int(channels[3]/2))
        self.ConvBlock1 = conv_block(ch_in=2*channels[3], ch_out=channels[3])
        
        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(2*channels[1])
        self.CA2 = ChannelAttention(2*channels[2])
        self.CA1 = ChannelAttention(2*channels[3])
        
        self.SA = SpatialAttention()
      
    def forward(self,x, skips):
    
        d4 = self.Conv_1x1(x)
        
        # CAM4
        d4 = self.CA4(d4)*d4
        d4 = self.SA(d4)*d4 
        d4 = self.ConvBlock4(d4)
        
        # upconv3
        d3 = self.Up3(d4)
        
        # AG3
        x3 = self.AG3(g=d3,x=skips[0])
        
        # Concat 3
        d3 = torch.cat((x3,d3),dim=1)
        
        # CAM3
        d3 = self.CA3(d3)*d3
        d3 = self.SA(d3)*d3        
        d3 = self.ConvBlock3(d3)
        
        # upconv2
        d2 = self.Up2(d3)
        
        # AG2
        x2 = self.AG2(g=d2,x=skips[1])
        
        # Concat 2
        d2 = torch.cat((x2,d2),dim=1)
        
        # CAM2
        d2 = self.CA2(d2)*d2
        d2 = self.SA(d2)*d2
        #print(d2.shape)
        d2 = self.ConvBlock2(d2)
        
        # upconv1
        d1 = self.Up1(d2)
        
        #print(skips[2])
        # AG1
        x1 = self.AG1(g=d1,x=skips[2])
        
        # Concat 1
        d1 = torch.cat((x1,d1),dim=1)
        
        # CAM1
        d1 = self.CA1(d1)*d1
        d1 = self.SA(d1)*d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1
        

class CASCADE_Add(nn.Module):
    def __init__(self, channels=[512,320,128,64]):
        super(CASCADE_Add,self).__init__()
        
        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])
	
        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=channels[2], ch_out=channels[2])
        
        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3],F_l=channels[3],F_int=int(channels[3]/2))
        self.ConvBlock1 = conv_block(ch_in=channels[3], ch_out=channels[3])
        
        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(channels[1])
        self.CA2 = ChannelAttention(channels[2])
        self.CA1 = ChannelAttention(channels[3])
        
        self.SA = SpatialAttention()
      
    def forward(self,x, skips):
    
        d4 = self.Conv_1x1(x)
        
        # CAM4
        d4 = self.CA4(d4)*d4
        d4 = self.SA(d4)*d4 
        d4 = self.ConvBlock4(d4)
        
        # upconv3
        d3 = self.Up3(d4)
        
        # AG3
        x3 = self.AG3(g=d3,x=skips[0])
        
        # aggregate 3
        d3 = d3 + x3
        
        # CAM3
        d3 = self.CA3(d3)*d3
        d3 = self.SA(d3)*d3        
        d3 = self.ConvBlock3(d3)
        
        # upconv2
        d2 = self.Up2(d3)
        
        # AG2
        x2 = self.AG2(g=d2,x=skips[1])
        
        # aggregate 2
        d2 = d2 + x2
        
        # CAM2
        d2 = self.CA2(d2)*d2
        d2 = self.SA(d2)*d2
        #print(d2.shape)
        d2 = self.ConvBlock2(d2)
        
        # upconv1
        d1 = self.Up1(d2)
        
        #print(skips[2])
        # AG1
        x1 = self.AG1(g=d1,x=skips[2])
        
        # aggregate 1
        d1 = d1 + x1
        
        # CAM1
        d1 = self.CA1(d1)*d1
        d1 = self.SA(d1)*d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1


class CASCADE_Add_SkipFusion(nn.Module):
    def __init__(self, channels=[512,320,128,64]):
        super(CASCADE_Add_SkipFusion,self).__init__()

        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])

        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.Fuse3 = SkipSelectiveFusion(F_g=channels[1],F_l=channels[1],F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.Fuse2 = SkipSelectiveFusion(F_g=channels[2],F_l=channels[2],F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=channels[2], ch_out=channels[2])

        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.Fuse1 = SkipSelectiveFusion(F_g=channels[3],F_l=channels[3],F_int=int(channels[3]/2))
        self.ConvBlock1 = conv_block(ch_in=channels[3], ch_out=channels[3])

        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(channels[1])
        self.CA2 = ChannelAttention(channels[2])
        self.CA1 = ChannelAttention(channels[3])

        self.SA = SpatialAttention()

    def forward(self,x, skips):

        d4 = self.Conv_1x1(x)

        # CAM4
        d4 = self.CA4(d4)*d4
        d4 = self.SA(d4)*d4
        d4 = self.ConvBlock4(d4)

        # upconv3 + selective skip fusion
        d3 = self.Up3(d4)
        d3 = self.Fuse3(g=d3,x=skips[0])

        # CAM3
        d3 = self.CA3(d3)*d3
        d3 = self.SA(d3)*d3
        d3 = self.ConvBlock3(d3)

        # upconv2 + selective skip fusion
        d2 = self.Up2(d3)
        d2 = self.Fuse2(g=d2,x=skips[1])

        # CAM2
        d2 = self.CA2(d2)*d2
        d2 = self.SA(d2)*d2
        d2 = self.ConvBlock2(d2)

        # upconv1 + selective skip fusion
        d1 = self.Up1(d2)
        d1 = self.Fuse1(g=d1,x=skips[2])

        # CAM1
        d1 = self.CA1(d1)*d1
        d1 = self.SA(d1)*d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1


class CASCADE_Add_SkipResidual(nn.Module):
    def __init__(self, channels=[512,320,128,64]):
        super(CASCADE_Add_SkipResidual,self).__init__()

        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])

        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=channels[2], ch_out=channels[2])

        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3],F_l=channels[3],F_int=int(channels[3]/2))
        self.ConvBlock1 = conv_block(ch_in=channels[3], ch_out=channels[3])

        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(channels[1])
        self.CA2 = ChannelAttention(channels[2])
        self.CA1 = ChannelAttention(channels[3])

        self.SA = SpatialAttention()

        self.C3 = SkipResidualAdapter(channels[1])
        self.C2 = SkipResidualAdapter(channels[2])
        self.C1 = SkipResidualAdapter(channels[3])

    def forward(self,x, skips):

        d4 = self.Conv_1x1(x)

        # CAM4
        d4 = self.CA4(d4)*d4
        d4 = self.SA(d4)*d4
        d4 = self.ConvBlock4(d4)

        # upconv3 + original AG addition + zero-init C side adapter
        d3 = self.Up3(d4)
        x3 = self.AG3(g=d3,x=skips[0])
        d3 = d3 + x3
        d3 = self.C3(d3, x3)

        # CAM3
        d3 = self.CA3(d3)*d3
        d3 = self.SA(d3)*d3
        d3 = self.ConvBlock3(d3)

        # upconv2 + original AG addition + zero-init C side adapter
        d2 = self.Up2(d3)
        x2 = self.AG2(g=d2,x=skips[1])
        d2 = d2 + x2
        d2 = self.C2(d2, x2)

        # CAM2
        d2 = self.CA2(d2)*d2
        d2 = self.SA(d2)*d2
        d2 = self.ConvBlock2(d2)

        # upconv1 + original AG addition + zero-init C side adapter
        d1 = self.Up1(d2)
        x1 = self.AG1(g=d1,x=skips[2])
        d1 = d1 + x1
        d1 = self.C1(d1, x1)

        # CAM1
        d1 = self.CA1(d1)*d1
        d1 = self.SA(d1)*d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1


class CASCADE_Add_SkipResidualC1(nn.Module):
    def __init__(self, channels=[512,320,128,64]):
        super(CASCADE_Add_SkipResidualC1,self).__init__()

        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])

        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=channels[2], ch_out=channels[2])

        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3],F_l=channels[3],F_int=int(channels[3]/2))
        self.ConvBlock1 = conv_block(ch_in=channels[3], ch_out=channels[3])

        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(channels[1])
        self.CA2 = ChannelAttention(channels[2])
        self.CA1 = ChannelAttention(channels[3])

        self.SA = SpatialAttention()
        self.C1 = BoundedSkipResidualAdapter(channels[3], max_scale=0.5)

    def forward(self,x, skips):

        d4 = self.Conv_1x1(x)

        # CAM4
        d4 = self.CA4(d4)*d4
        d4 = self.SA(d4)*d4
        d4 = self.ConvBlock4(d4)

        # upconv3 + original AG addition
        d3 = self.Up3(d4)
        x3 = self.AG3(g=d3,x=skips[0])
        d3 = d3 + x3

        # CAM3
        d3 = self.CA3(d3)*d3
        d3 = self.SA(d3)*d3
        d3 = self.ConvBlock3(d3)

        # upconv2 + original AG addition
        d2 = self.Up2(d3)
        x2 = self.AG2(g=d2,x=skips[1])
        d2 = d2 + x2

        # CAM2
        d2 = self.CA2(d2)*d2
        d2 = self.SA(d2)*d2
        d2 = self.ConvBlock2(d2)

        # upconv1 + original AG addition + bounded high-resolution C adapter
        d1 = self.Up1(d2)
        x1 = self.AG1(g=d1,x=skips[2])
        d1 = d1 + x1
        d1 = self.C1(d1, x1)

        # CAM1
        d1 = self.CA1(d1)*d1
        d1 = self.SA(d1)*d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1


class CASCADE_Add_JointChannel(nn.Module):
    """AB-compatible decoder with spatial AG followed by joint channel recalibration.

    The existing decoder modules are reused instead of reinitialized. This keeps all
    common AB parameters identical for a fixed seed; only C1/C2/C3 are newly added.
    """

    def __init__(self, base_decoder, channels=[512, 320, 128, 64]):
        super(CASCADE_Add_JointChannel, self).__init__()
        for name, module in base_decoder.named_children():
            setattr(self, name, module)
        # Keep the global RNG stream aligned with AB so its data shuffle is unchanged.
        with torch.random.fork_rng(devices=[]):
            self.C3 = JointChannelRecalibration(channels[1])
            self.C2 = JointChannelRecalibration(channels[2])
            self.C1 = JointChannelRecalibration(channels[3])

    def forward(self, x, skips):
        d4 = self.Conv_1x1(x)
        d4 = self.CA4(d4) * d4
        d4 = self.SA(d4) * d4
        d4 = self.ConvBlock4(d4)

        d3 = self.Up3(d4)
        x3 = self.AG3(g=d3, x=skips[0])
        d3 = self.C3(d3, x3)
        d3 = self.CA3(d3) * d3
        d3 = self.SA(d3) * d3
        d3 = self.ConvBlock3(d3)

        d2 = self.Up2(d3)
        x2 = self.AG2(g=d2, x=skips[1])
        d2 = self.C2(d2, x2)
        d2 = self.CA2(d2) * d2
        d2 = self.SA(d2) * d2
        d2 = self.ConvBlock2(d2)

        d1 = self.Up1(d2)
        x1 = self.AG1(g=d1, x=skips[2])
        d1 = self.C1(d1, x1)
        d1 = self.CA1(d1) * d1
        d1 = self.SA(d1) * d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1


class CASCADE_Add_JointChannelC1(nn.Module):
    """AB-compatible decoder with channel recalibration only at the finest skip."""

    def __init__(self, base_decoder, channels=[512, 320, 128, 64]):
        super(CASCADE_Add_JointChannelC1, self).__init__()
        for name, module in base_decoder.named_children():
            setattr(self, name, module)
        with torch.random.fork_rng(devices=[]):
            self.C1 = JointChannelRecalibration(channels[3])

    def forward(self, x, skips):
        d4 = self.Conv_1x1(x)
        d4 = self.CA4(d4) * d4
        d4 = self.SA(d4) * d4
        d4 = self.ConvBlock4(d4)

        d3 = self.Up3(d4)
        d3 = d3 + self.AG3(g=d3, x=skips[0])
        d3 = self.CA3(d3) * d3
        d3 = self.SA(d3) * d3
        d3 = self.ConvBlock3(d3)

        d2 = self.Up2(d3)
        d2 = d2 + self.AG2(g=d2, x=skips[1])
        d2 = self.CA2(d2) * d2
        d2 = self.SA(d2) * d2
        d2 = self.ConvBlock2(d2)

        d1 = self.Up1(d2)
        x1 = self.AG1(g=d1, x=skips[2])
        d1 = self.C1(d1, x1)
        d1 = self.CA1(d1) * d1
        d1 = self.SA(d1) * d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1


class CASCADE_Add_JointChannelC23(nn.Module):
    """AB-compatible channel recalibration on the /16 and /8 semantic skips."""

    def __init__(self, base_decoder, channels=[512, 320, 128, 64], max_delta=0.5):
        super(CASCADE_Add_JointChannelC23, self).__init__()
        for name, module in base_decoder.named_children():
            setattr(self, name, module)
        with torch.random.fork_rng(devices=[]):
            self.C3 = JointChannelRecalibration(channels[1], max_delta=max_delta)
            self.C2 = JointChannelRecalibration(channels[2], max_delta=max_delta)

    def forward(self, x, skips):
        d4 = self.Conv_1x1(x)
        d4 = self.CA4(d4) * d4
        d4 = self.SA(d4) * d4
        d4 = self.ConvBlock4(d4)

        d3 = self.Up3(d4)
        x3 = self.AG3(g=d3, x=skips[0])
        d3 = self.C3(d3, x3)
        d3 = self.CA3(d3) * d3
        d3 = self.SA(d3) * d3
        d3 = self.ConvBlock3(d3)

        d2 = self.Up2(d3)
        x2 = self.AG2(g=d2, x=skips[1])
        d2 = self.C2(d2, x2)
        d2 = self.CA2(d2) * d2
        d2 = self.SA(d2) * d2
        d2 = self.ConvBlock2(d2)

        d1 = self.Up1(d2)
        d1 = d1 + self.AG1(g=d1, x=skips[2])
        d1 = self.CA1(d1) * d1
        d1 = self.SA(d1) * d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1


class CASCADE_Add_JointChannelC3(nn.Module):
    """AB-compatible channel recalibration only on the deepest semantic skip."""

    def __init__(self, base_decoder, channels=[512, 320, 128, 64]):
        super(CASCADE_Add_JointChannelC3, self).__init__()
        for name, module in base_decoder.named_children():
            setattr(self, name, module)
        with torch.random.fork_rng(devices=[]):
            self.C3 = JointChannelRecalibration(channels[1])

    def forward(self, x, skips):
        d4 = self.Conv_1x1(x)
        d4 = self.CA4(d4) * d4
        d4 = self.SA(d4) * d4
        d4 = self.ConvBlock4(d4)

        d3 = self.Up3(d4)
        x3 = self.AG3(g=d3, x=skips[0])
        d3 = self.C3(d3, x3)
        d3 = self.CA3(d3) * d3
        d3 = self.SA(d3) * d3
        d3 = self.ConvBlock3(d3)

        d2 = self.Up2(d3)
        d2 = d2 + self.AG2(g=d2, x=skips[1])
        d2 = self.CA2(d2) * d2
        d2 = self.SA(d2) * d2
        d2 = self.ConvBlock2(d2)

        d1 = self.Up1(d2)
        d1 = d1 + self.AG1(g=d1, x=skips[2])
        d1 = self.CA1(d1) * d1
        d1 = self.SA(d1) * d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1


class CASCADE_Add_JointChannelC2(nn.Module):
    """AB-compatible channel recalibration only on the middle semantic skip."""

    def __init__(self, base_decoder, channels=[512, 320, 128, 64], max_delta=0.5):
        super(CASCADE_Add_JointChannelC2, self).__init__()
        for name, module in base_decoder.named_children():
            setattr(self, name, module)
        with torch.random.fork_rng(devices=[]):
            self.C2 = JointChannelRecalibration(channels[2], max_delta=max_delta)

    def forward(self, x, skips):
        d4 = self.Conv_1x1(x)
        d4 = self.CA4(d4) * d4
        d4 = self.SA(d4) * d4
        d4 = self.ConvBlock4(d4)

        d3 = self.Up3(d4)
        d3 = d3 + self.AG3(g=d3, x=skips[0])
        d3 = self.CA3(d3) * d3
        d3 = self.SA(d3) * d3
        d3 = self.ConvBlock3(d3)

        d2 = self.Up2(d3)
        x2 = self.AG2(g=d2, x=skips[1])
        d2 = self.C2(d2, x2)
        d2 = self.CA2(d2) * d2
        d2 = self.SA(d2) * d2
        d2 = self.ConvBlock2(d2)

        d1 = self.Up1(d2)
        d1 = d1 + self.AG1(g=d1, x=skips[2])
        d1 = self.CA1(d1) * d1
        d1 = self.SA(d1) * d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1


class CASCADE_Add_ComplementaryC1(nn.Module):
    """AB-compatible decoder with boundary-aware complementary finest-skip fusion."""

    def __init__(self, base_decoder, channels=[512, 320, 128, 64]):
        super(CASCADE_Add_ComplementaryC1, self).__init__()
        for name, module in base_decoder.named_children():
            setattr(self, name, module)
        with torch.random.fork_rng(devices=[]):
            self.C1 = ComplementaryChannelFusion(channels[3])

    def forward(self, x, skips):
        d4 = self.Conv_1x1(x)
        d4 = self.CA4(d4) * d4
        d4 = self.SA(d4) * d4
        d4 = self.ConvBlock4(d4)

        d3 = self.Up3(d4)
        d3 = d3 + self.AG3(g=d3, x=skips[0])
        d3 = self.CA3(d3) * d3
        d3 = self.SA(d3) * d3
        d3 = self.ConvBlock3(d3)

        d2 = self.Up2(d3)
        d2 = d2 + self.AG2(g=d2, x=skips[1])
        d2 = self.CA2(d2) * d2
        d2 = self.SA(d2) * d2
        d2 = self.ConvBlock2(d2)

        d1 = self.Up1(d2)
        x1 = self.AG1(g=d1, x=skips[2])
        d1 = self.C1(d1, x1)
        d1 = self.CA1(d1) * d1
        d1 = self.SA(d1) * d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1


class CASCADE_Add_ComplementaryC12(nn.Module):
    """AB-compatible boundary fusion at the two highest-resolution skips."""

    def __init__(self, base_decoder, channels=[512, 320, 128, 64]):
        super(CASCADE_Add_ComplementaryC12, self).__init__()
        for name, module in base_decoder.named_children():
            setattr(self, name, module)
        with torch.random.fork_rng(devices=[]):
            self.C2 = ComplementaryChannelFusion(channels[2])
            self.C1 = ComplementaryChannelFusion(channels[3])

    def forward(self, x, skips):
        d4 = self.Conv_1x1(x)
        d4 = self.CA4(d4) * d4
        d4 = self.SA(d4) * d4
        d4 = self.ConvBlock4(d4)

        d3 = self.Up3(d4)
        d3 = d3 + self.AG3(g=d3, x=skips[0])
        d3 = self.CA3(d3) * d3
        d3 = self.SA(d3) * d3
        d3 = self.ConvBlock3(d3)

        d2 = self.Up2(d3)
        x2 = self.AG2(g=d2, x=skips[1])
        d2 = self.C2(d2, x2)
        d2 = self.CA2(d2) * d2
        d2 = self.SA(d2) * d2
        d2 = self.ConvBlock2(d2)

        d1 = self.Up1(d2)
        x1 = self.AG1(g=d1, x=skips[2])
        d1 = self.C1(d1, x1)
        d1 = self.CA1(d1) * d1
        d1 = self.SA(d1) * d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1


class CASCADE_Add_HybridC12(nn.Module):
    """Scale-specialized C: semantic recalibration at C2 and boundary fusion at C1."""

    def __init__(self, base_decoder, channels=[512, 320, 128, 64]):
        super(CASCADE_Add_HybridC12, self).__init__()
        for name, module in base_decoder.named_children():
            setattr(self, name, module)
        with torch.random.fork_rng(devices=[]):
            self.C2 = JointChannelRecalibration(channels[2], max_delta=0.25)
            self.C1 = ComplementaryChannelFusion(channels[3], max_delta=0.25)

    def forward(self, x, skips):
        d4 = self.Conv_1x1(x)
        d4 = self.CA4(d4) * d4
        d4 = self.SA(d4) * d4
        d4 = self.ConvBlock4(d4)

        d3 = self.Up3(d4)
        d3 = d3 + self.AG3(g=d3, x=skips[0])
        d3 = self.CA3(d3) * d3
        d3 = self.SA(d3) * d3
        d3 = self.ConvBlock3(d3)

        d2 = self.Up2(d3)
        x2 = self.AG2(g=d2, x=skips[1])
        d2 = self.C2(d2, x2)
        d2 = self.CA2(d2) * d2
        d2 = self.SA(d2) * d2
        d2 = self.ConvBlock2(d2)

        d1 = self.Up1(d2)
        x1 = self.AG1(g=d1, x=skips[2])
        d1 = self.C1(d1, x1)
        d1 = self.CA1(d1) * d1
        d1 = self.SA(d1) * d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1
