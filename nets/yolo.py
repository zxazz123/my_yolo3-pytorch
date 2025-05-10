from collections import OrderedDict

import torch
import torch.nn as nn

from nets.darknet import darknet53

def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("mish", nn.Mish()),
    ]))

#------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
#------------------------------------------------------------------------#
# def make_last_layers(filters_list, in_filters, out_filter):
#     m = nn.Sequential(
#         conv2d(in_filters, filters_list[0], 1),
#         conv2d(filters_list[0], filters_list[1], 3),
#         conv2d(filters_list[1], filters_list[0], 1),
#         conv2d(filters_list[0], filters_list[1], 3),
#         conv2d(filters_list[1], filters_list[0], 1),
#         conv2d(filters_list[0], filters_list[1], 3),
#         nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
#     )
#     return m
class Spp(nn.Module):
    def __init__(self,in_filters):
        super(Spp, self).__init__()
        self.conv1 = conv2d(in_filters,in_filters//2,1)
        self.conv2 = conv2d(in_filters//2,in_filters,3)
        self.conv3 = conv2d(in_filters,in_filters//2,1)

        kernel_size = [5,9,13]
        padding = kernel_size[0] // 2  # 确保输出尺寸不变
        self.max_pool1 = nn.MaxPool2d(kernel_size=kernel_size[0], stride=1, padding=padding)

        padding = kernel_size[1] // 2  # 确保输出尺寸不变
        self.max_pool2 = nn.MaxPool2d(kernel_size=kernel_size[1], stride=1, padding=padding)

        padding = kernel_size[2] // 2  # 确保输出尺寸不变
        self.max_pool3 = nn.MaxPool2d(kernel_size=kernel_size[2], stride=1, padding=padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x1 = self.max_pool1(x)
        x2 = self.max_pool2(x)
        x3 = self.max_pool3(x)
        out = torch.cat([x,x1,x2,x3],1)
        return out

class Fpn(nn.Module):
    def __init__(self,in_c):
        super(Fpn, self).__init__()
        c1 = nn.Sequential(OrderedDict([
            ("conv1", conv2d(in_c[0],512,1)),
            ("conv2", conv2d(512,1024,3)),
            ("conv3", conv2d(1024,512,1)),
            ("conv4", conv2d(512,256,1)),
            ]))
        c2 = nn.Sequential(OrderedDict([
            ("conv1", conv2d(512,256,1)),
            ("conv2", conv2d(256,512,3)),
            ("conv3", conv2d(512,256,1)),
            ("conv4", conv2d(256,512,3)),
            ("conv5", conv2d(512,256,1)),
            ("conv6", conv2d(256,128,1))
            ]))
        self.c3 = conv2d(in_c[1],in_c[1]//2,1)
        self.c4 = conv2d(in_c[2],in_c[2]//2,1)

        self.conv_before_us = nn.ModuleList([c1,c2])
        self.up_sample = nn.Upsample(scale_factor=2)
    def forward(self, x0,x1,x2):
        out0 = self.conv_before_us[0][:3](x0)
        con1 = self.conv_before_us[0][3](out0)
        con1 = self.up_sample(con1)
        con2 = self.c3(x1)
        x = torch.cat([con1,con2],1)

        out1 = self.conv_before_us[1][:5](x)
        x = self.conv_before_us[1][5](out1)
        con1 = self.up_sample(x)
        con2 = self.c4(x2)
        out2 = torch.cat([con1,con2],1)

        return out0,out1,out2
    
class Head(nn.Module):
    def __init__(self,in_c,anchors_mask,num_classes):
        super(Head, self).__init__()
        self.five_conv1 = nn.Sequential(OrderedDict([
            ("conv1", conv2d(in_c[2],128,1)),
            ("conv2", conv2d(128,256,3)),
            ("conv3", conv2d(256,128,1)),
            ("conv4", conv2d(128,256,3)),
            ("conv5", conv2d(256,128,1))
            ]))
        self.pre1 = nn.Sequential(OrderedDict([
            ("conv1", conv2d(128,256,3)),
            ("conv2", nn.Conv2d(256, (len(anchors_mask[2]) * (num_classes + 5)), kernel_size=1, stride=1, padding=0, bias=True))
            ]))
        
        self.ds1 = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)),
            ("bn", nn.BatchNorm2d(256)),
            ("mish", nn.Mish())
            ]))
        
        self.five_conv2 = nn.Sequential(OrderedDict([
            ("conv1", conv2d(512,256,1)),
            ("conv2", conv2d(256,512,3)),
            ("conv3", conv2d(512,256,1)),
            ("conv4", conv2d(256,512,3)),
            ("conv5", conv2d(512,256,1))
            ]))
        self.pre2 = nn.Sequential(OrderedDict([
            ("conv1", conv2d(256,512,3)),
            ("conv2", nn.Conv2d(512, (len(anchors_mask[1]) * (num_classes + 5)), kernel_size=1, stride=1, padding=0, bias=True))
            ]))
        
        self.ds2 = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)),
            ("bn", nn.BatchNorm2d(512)),
            ("mish", nn.Mish())
            ]))
        
        self.five_conv3 = nn.Sequential(OrderedDict([
            ("conv1", conv2d(1024,512,1)),
            ("conv2", conv2d(512,1024,3)),
            ("conv3", conv2d(1024,512,1)),
            ("conv4", conv2d(512,1024,3)),
            ("conv5", conv2d(1024,512,1))
            ]))
        self.pre3 = nn.Sequential(OrderedDict([
            ("conv1", conv2d(512,1024,3)),
            ("conv2", nn.Conv2d(1024, (len(anchors_mask[0]) * (num_classes + 5)), kernel_size=1, stride=1, padding=0, bias=True))
            ]))
        
    def forward(self, x0,x1,x2):
        out2_branch = self.five_conv1(x2)
        out2 = self.pre1(out2_branch)

        x = torch.cat([x1,self.ds1(out2_branch)],1)
        out1_branch = self.five_conv2(x)
        out1 = self.pre2(out1_branch)

        x = torch.cat([x0,self.ds2(out1_branch)],1)
        x = self.five_conv3(x)
        out0 = self.pre3(x)
        return out0,out1,out2
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes ,pretrained = False):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#   
        #   生成darknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone = darknet53()
        if pretrained:
            self.backbone.load_state_dict(torch.load("model_data/darknet53_backbone_weights.pth"))

        #---------------------------------------------------#
        #   out_filters : [64, 128, 256, 512, 1024]
        #---------------------------------------------------#
        out_filters = self.backbone.layers_out_filters

        #------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        #------------------------------------------------------------------------#
        #self.last_layer0            = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))

        #self.last_layer1_conv       = conv2d(512, 256, 1)
        #self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        #self.last_layer1            = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))

        #self.last_layer2_conv       = conv2d(256, 128, 1)
        #self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        #self.last_layer2            = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

        self.spp_layer = Spp(out_filters[-1])
        self.fpn = Fpn((2048,out_filters[-2],out_filters[-3]))
        self.head = Head((512,256,256),anchors_mask,num_classes)
    def forward(self, x):
        #---------------------------------------------------#   
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256；26,26,512；13,13,1024
        #---------------------------------------------------#
        
        x2, x1, x0 = self.backbone(x)
        x0 = self.spp_layer(x0)
        out0,out1,out2 = self.fpn(x0,x1,x2)
        out0,out1,out2 = self.head(out0,out1,out2)
        
        #---------------------------------------------------#
        #   第一个特征层
        #   out0 = (batch_size,255,13,13)
        #---------------------------------------------------#
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        #out0_branch = self.last_layer0[:5](x0)
        #out0        = self.last_layer0[5:](out0_branch)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        #x1_in = self.last_layer1_conv(out0_branch)
        #x1_in = self.last_layer1_upsample(x1_in)

        # 26,26,256 + 26,26,512 -> 26,26,768
        #x1_in = torch.cat([x1_in, x1], 1)
        #---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,255,26,26)
        #---------------------------------------------------#
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        #out1_branch = self.last_layer1[:5](x1_in)
        #out1        = self.last_layer1[5:](out1_branch)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        #x2_in = self.last_layer2_conv(out1_branch)
        #x2_in = self.last_layer2_upsample(x2_in)

        # 52,52,128 + 52,52,256 -> 52,52,384
        #x2_in = torch.cat([x2_in, x2], 1)
        #---------------------------------------------------#
        #   第一个特征层
        #   out3 = (batch_size,255,52,52)
        #---------------------------------------------------#
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        #out2 = self.last_layer2(x2_in)
        return out0, out1, out2