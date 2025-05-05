import math
from collections import OrderedDict

import torch.nn as nn
import torch

#---------------------------------------------------------------------#
#   残差结构
#   利用一个1x1卷积下降通道数，然后利用一个3x3卷积提取特征并且上升通道数
#   最后接上一个残差边
#---------------------------------------------------------------------#
#_开头的变量为新增,如self._conv1
class BasicBlock(nn.Module):
    def __init__(self, planes):
        super(BasicBlock, self).__init__()
        #如果是Darknet53分块1
        if planes[0] == 64 and planes[1] == 64:
            self.conv1  = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1    = nn.BatchNorm2d(32)
        else:
            self.conv1  = nn.Conv2d(planes[1], planes[1], kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1    = nn.BatchNorm2d(planes[1])

        self.mish1  = nn.Mish()

        if planes[0] == 64 and planes[1] == 64:
            self.conv2  = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2    = nn.BatchNorm2d(64)
        else:
            self.conv2  = nn.Conv2d(planes[1], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2    = nn.BatchNorm2d(planes[1])
        self.mish2  = nn.Mish()

        
    def forward(self, x):
        #residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.mish1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.mish2(out)

        out += x

        return out

class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        # 416,416,3 -> 416,416,32
        self.conv1  = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(self.inplanes)
        self.mish1  = nn.Mish()
        
        self.layers_out_filters = [64, 128, 256, 512, 1024]
        self.dssq = []
        self.con2sq = []
        self.conv_after_concat_sq = []

        # 下采样，步长为2，卷积核大小为3和用于通道连接的1x1卷积
        for i in range(len(self.layers_out_filters)):
            d = []
            c = []
            cac = []

            ii = self.inplanes if i == 0 else self.layers_out_filters[i-1]
            d.append(('ds'+str(i+1),nn.Conv2d(ii, self.layers_out_filters[i], kernel_size=3, stride=2, padding=1, bias=False)))
            d.append(('dsbn'+str(i+1),nn.BatchNorm2d(self.layers_out_filters[i])))
            d.append(('dsmish'+str(i+1),nn.Mish()))

            ii = self.layers_out_filters[0] if i == 0 else self.layers_out_filters[i-1]
            c.append(('con2'+str(i+1),nn.Conv2d(self.layers_out_filters[i], ii, kernel_size=1, stride=1, padding=0, bias=False)))
            c.append(('con2bn'+str(i+1),nn.BatchNorm2d(ii)))
            c.append(('con2mish'+str(i+1),nn.Mish()))

            ii = self.layers_out_filters[0]*2 if i == 0 else self.layers_out_filters[i]
            cac.append(('conv_after_concat'+str(i+1),nn.Conv2d(ii, self.layers_out_filters[i], kernel_size=1, stride=1, padding=0, bias=False)))
            cac.append(('conv_after_concat_bn'+str(i+1),nn.BatchNorm2d(self.layers_out_filters[i])))
            cac.append(('conv_after_concat_mish'+str(i+1),nn.Mish()))
            
            self.dssq.append(nn.Sequential(OrderedDict(d)))
            self.con2sq.append(nn.Sequential(OrderedDict(c)))
            self.conv_after_concat_sq.append(nn.Sequential(OrderedDict(cac)))

        self.dssq = nn.ModuleList(self.dssq)
        self.con2sq = nn.ModuleList(self.con2sq)
        self.conv_after_concat_sq = nn.ModuleList(self.conv_after_concat_sq)

        # 416,416,32 -> 208,208,64 -> 208,208,64
        self.layer1 = self._make_layer([64, 64], layers[0])
        # 208,208,64 -> 104,104,128 -> 104,104,64
        self.layer2 = self._make_layer([128, 64], layers[1])
        # 104,104,128 -> 52,52,256
        self.layer3 = self._make_layer([256, 128], layers[2])
        # 52,52,256 -> 26,26,512
        self.layer4 = self._make_layer([512, 256], layers[3])
        # 26,26,512 -> 13,13,1024
        self.layer5 = self._make_layer([1024, 512], layers[4])

        

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #---------------------------------------------------------------------#
    #   在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
    #   然后进行残差结构的堆叠
    #---------------------------------------------------------------------#
    def _make_layer(self, planes, blocks):
        layers = []
        
        layers.append(("ds_conv_1" , nn.Conv2d(planes[0], planes[1], kernel_size=1, stride=1, padding=0, bias=False)))
        layers.append(("ds_bn_1",nn.BatchNorm2d(planes[1])))
        layers.append(("ds_mish_1",nn.Mish()))
        self.inplanes = planes[1]

        # 加入残差结构
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(planes)))
        
        layers.append(("ds_conv_2" , nn.Conv2d(planes[1], planes[1], kernel_size=1, stride=1, padding=0, bias=False)))
        layers.append(("ds_bn_2",nn.BatchNorm2d(planes[1])))
        layers.append(("ds_mish_2",nn.Mish()))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.mish1(x)

        x = self.dssq[0](x)
        con1 = self.layer1(x)
        con2 = self.con2sq[0](x)
        x = torch.cat((con1, con2), dim=1)
        x = self.conv_after_concat_sq[0](x)

        x = self.dssq[1](x)
        con1 = self.layer2(x)
        con2 = self.con2sq[1](x)
        x = torch.cat((con1, con2), dim=1)
        x = self.conv_after_concat_sq[1](x)

        x = self.dssq[2](x)
        con1 = self.layer3(x)
        con2 = self.con2sq[2](x)
        out3 = torch.cat((con1, con2), dim=1)
        out3 = self.conv_after_concat_sq[2](out3)

        x = self.dssq[3](out3)
        con1 = self.layer4(x)
        con2 = self.con2sq[3](x)
        out4 = torch.cat((con1, con2), dim=1)
        out4 = self.conv_after_concat_sq[3](out4)

        x = self.dssq[4](out4)
        con1 = self.layer5(x)
        con2 = self.con2sq[4](x)
        out5 = torch.cat((con1, con2), dim=1)
        out5 = self.conv_after_concat_sq[4](out5)

        return out3, out4, out5

def darknet53():
    model = DarkNet([1, 2, 8, 8, 4])
    return model
