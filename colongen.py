import torch
import torch.nn as nn
import torch.nn.functional as F
from pvtv2 import pvt_v2_b2
from davit import DaViT
import os

from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MSFM(nn.Module):
    def __init__(self, channel):
        super(MSFM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

        self.asppo1 = ASPPout(32)
        self.asppo2 = ASPPout(32)
        self.asppo3 = ASPPout(32)

    def forward(self, x1, x2, x3):
        x1=self.asppo1(x1)
        x2=self.asppo2(x2)
        x3=self.asppo3(x3)
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1
        

class MSFM2(nn.Module):
    def __init__(self, channel):
        super(MSFM2, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.Upsample(scale_factor=.5, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

        self.asppo1 = ASPPout(64)
        self.asppo2 = ASPPout(64)
        self.asppo3 = ASPPout(64)

    def forward(self, x1, x2, x3):
        x1=self.asppo1(x1)
        x2=self.asppo2(x2)
        x3=self.asppo3(x3)
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        X3_2 = torch.cat((self.down(x3_1), self.conv_upsample5(x2_2)) , 1)

        #x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(X3_2)

        x1 = self.conv4(x3_2)

        return x1



class MSFM3(nn.Module):
    def __init__(self, channel):
        super(MSFM3, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)
        
        
        
        self.asppo1 = ASPPout(32)
        self.asppo2 = ASPPout(32)
        self.asppo3 = ASPPout(32)

    def forward(self, x1, x2, x3):
        x1=self.asppo1(x1)
        x2=self.asppo2(x2)
        x3=self.asppo3(x3)
        
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
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






################

class ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)




class ASPPout(nn.Module):
    def __init__(self, planes):
        super(ASPPout, self).__init__()

        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        
        dilations = [1, 2, 3, 3]
        
        self.aspp1 = ASPPModule(planes, 64, 3, padding=1, dilation=dilations[0], BatchNorm=nn.BatchNorm2d)
        self.aspp2 = ASPPModule(planes, 64, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=nn.BatchNorm2d)
        self.aspp3 = ASPPModule(planes, 64, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=nn.BatchNorm2d)
        self.aspp4 = ASPPModule(planes, 64, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=nn.BatchNorm2d)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(planes, 64, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(64),
                                             nn.ReLU())
                                             
        self.conv1 = nn.Conv2d(320, planes, 1, bias=False)


    def forward(self, x):

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn(x)
        return self.relu(x)
#################


class ColonGen(nn.Module):
    def __init__(self, channel=32):
        super(ColonGen, self).__init__()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone = pvt_v2_b2() 
        path = 'models/pvt_v2_b2.pth'
        save_model = torch.load(path, map_location=device)

        path2 = 'models/model_best.pth.tar'
        save_davit = torch.load(path2, map_location=device)

        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.davit = DaViT()
        self.davit.load_state_dict(save_davit['state_dict'], strict= False)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.MSFM = MSFM(channel)
        self.MSFM2 = MSFM2(2*channel)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        
        self.ca2 = ChannelAttention(96)
        self.sa2 = SpatialAttention()
        
        self.ca3 = ChannelAttention(128)
        self.sa3 = SpatialAttention()
        
        self.ca4 = ChannelAttention(192)
        self.sa4 = SpatialAttention()
        
        self.MSFM3 = MSFM3(channel)
        
        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_MSFM3 = nn.Conv2d(channel, 1, 1)
        self.out_MSFM = nn.Conv2d(channel, 1, 1)
        self.out_MSFM2 = nn.Conv2d(2*channel, 1, 1)
        
    
        self.conv_backcat2 = BasicConv2d(192, 2*channel, 1)
        self.conv_backcat3 = BasicConv2d(384, 2*channel, 1)
        self.conv_backcat4 = BasicConv2d(768, 2*channel, 1)

        self.conv_y = BasicConv2d(64, channel, 1)
        self.conv_MSFM3 = BasicConv2d(64, channel, 1)
        
        self.se_feature_conv = BasicConv2d(160, 2*channel, 1)
        
        
        
    def forward(self, x):

        davit_x = self.davit(x)
        y1 = davit_x[0]
        y2 = davit_x[1]
        y3 = davit_x[2]
        y4 = davit_x[3]

        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        y2 = self.ca4(y2) * y2 
        y2 = self.sa4(y2) * y2 

        x2 = self.ca3(x2) * x2 
        x2 = self.sa3(x2) * x2 

        x11 = self.ca(x1) * x1
        se_feature = self.sa(x11) * x11
        
        y11 = self.ca2(y1) * y1 
        se_feature_y1 = self.sa2(y11) * y11  
  

        y2 = self.conv_backcat2(y2)
        y3 = self.conv_backcat3(y3)
        y4 = self.conv_backcat4(y4)
        
        y_out = self.MSFM2(y4, y3, y2)

        
        se_feature = torch.cat([se_feature, se_feature_y1],1)
        
        se_feature = self.se_feature_conv(se_feature)

        x2_t = self.Translayer2_1(x2)  
        x3_t = self.Translayer3_1(x3)  
        x4_t = self.Translayer4_1(x4)  
        MSFM_feature = self.MSFM(x4_t, x3_t, x2_t)

        MSFM3_feature = self.MSFM3(self.conv_y(y_out), MSFM_feature, self.conv_MSFM3(se_feature))




        prediction0 = self.out_MSFM2(y_out)
        prediction1 = self.out_MSFM(MSFM_feature)
        prediction2 = self.out_MSFM3(MSFM3_feature)

        prediction0_16 = F.interpolate(prediction0, scale_factor=16, mode='bilinear') 
        prediction1_8 = F.interpolate(prediction1, scale_factor=8, mode='bilinear') 
        prediction2_4 = F.interpolate(prediction2, scale_factor=4, mode='bilinear')  
        return prediction0_16, prediction1_8, prediction2_4
        