# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import imp

# import __init__ as booger
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResContextBlock_Circular(nn.Module):

    def __init__(self, in_filters, out_filters):
        print("ResContextBlock_Circular")
        super(ResContextBlock_Circular, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        # Added Padding
        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3), padding=(1,0))
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        # Added Padding
        self.conv3 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=(2,1))
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)


    def forward(self, x):

        # x = F.pad(x,(1,1,0,0),mode = 'circular')
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        # Added Circular Padding
        conv2_inp = F.pad(shortcut,(1,1,0,0),mode = 'circular')
        resA = self.conv2(conv2_inp)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)
        
        # Added Circular Padding
        conv3_inp = F.pad(resA1,(1,1,0,0),mode = 'circular')
        resA = self.conv3(conv3_inp)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        output = shortcut + resA2
        return output


class ResBlock_Circular(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        print("ResBlock_Circular")
        super(ResBlock_Circular, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        # Changed padding to accomodate circular padding
        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3,3), padding=(1,0))
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        # Changed padding to accomodate circular padding
        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3,3),dilation=2, padding=(2,1))
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        # Changed padding to accomodate circular padding
        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=(1,0))
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        # No Circular Padding because 1x1 convolutions
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        # Added Circular Padding
        x = F.pad(x,(1,1,0,0),mode = 'circular')
        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        # Added Circular Padding
        inp_conv3 = F.pad(resA1,(1,1,0,0),mode = 'circular')
        resA = self.conv3(inp_conv3)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        # Added Circular Padding
        inp_conv4 = F.pad(resA2,(1,1,0,0),mode = 'circular')
        resA = self.conv4(inp_conv4)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        # No Circular padding for concact 
        concat = torch.cat((resA1,resA2,resA3),dim=1)
        
        # No Circular Padding because 1x1 convolutions
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)        
        resA = shortcut + resA

        # No change to pooling and dropout
        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB


class UpBlock_Circular(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True):
        print("UpBlock_Circular")
        super(UpBlock_Circular, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        # Changed padding to accomodate circular padding
        self.conv1 = nn.Conv2d(in_filters//4 + 2*out_filters, out_filters, (3,3), padding=(1,0))
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        # Changed padding to accomodate circular padding
        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=(2,1))
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        # Changed padding to accomodate circular padding
        self.conv3 = nn.Conv2d(out_filters, out_filters, (2,2), dilation=2,padding=(1,0))
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        # Changed padding to accomodate circular padding
        self.conv4 = nn.Conv2d(out_filters*3,out_filters,kernel_size=(1,1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip, pd=False):

        # No change to Pixel Shuffle
        upA = nn.PixelShuffle(2)(x)
        if pd:
            upA = upA[:,:,:,:-1]
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA,skip),dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        # Added Circular Padding
        inp_conv1 = F.pad(upB,(1,1,0,0),mode = 'circular')
        upE = self.conv1(inp_conv1)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)
        
        # Added Circular Padding
        inp_conv2 = F.pad(upE1,(1,1,0,0),mode = 'circular')
        upE = self.conv2(inp_conv2)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)
        
        # Added Circular Padding
        inp_conv3 = F.pad(upE2,(1,1,0,0),mode = 'circular')
        upE = self.conv3(inp_conv3)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        # No Circular padding for concact 
        concat = torch.cat((upE1,upE2,upE3),dim=1)
        
        # No Circular Padding because 1x1 convolutions
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)

        # No Circular Padding for dropout 
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE


class SalsaNext_Circular(nn.Module):
    def __init__(self, nclasses):
        print("\n\nSalsaNext : Circular\n\n")
        super(SalsaNext_Circular, self).__init__()
        self.nclasses = nclasses

        # Chnaged ResContextBlock to ResContextBlock_Circular
        self.downCntx = ResContextBlock_Circular(32, 32)
        self.downCntx2 = ResContextBlock_Circular(32, 32)
        self.downCntx3 = ResContextBlock_Circular(32, 32)

        # Changed ResBlock to ResBlock_Circular
        self.resBlock1 = ResBlock_Circular(32, 2 * 32, 0.2, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock_Circular(2 * 32, 2 * 2 * 32, 0.2, pooling=True)
        self.resBlock3 = ResBlock_Circular(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock4 = ResBlock_Circular(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock5 = ResBlock_Circular(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)

        # Chnaged UpBlock to UpBlock_Circular
        self.upBlock1 = UpBlock_Circular(2 * 4 * 32, 4 * 32, 0.2)
        self.upBlock2 = UpBlock_Circular(4 * 32, 4 * 32, 0.2)
        self.upBlock3 = UpBlock_Circular(4 * 32, 2 * 32, 0.2)
        self.upBlock4 = UpBlock_Circular(2 * 32, 32, 0.2, drop_out=False)

        self.logits = nn.Conv2d(32, nclasses, kernel_size=(1, 1))

    def forward(self, x):

        downCntx = self.downCntx(x)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)
        
        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down5c = self.resBlock5(down3c)

        up4e = self.upBlock1(down5c,down3b,True)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)

        # No Circular Padding becasue 1x1 Convolutions
        x = self.logits(up1e)

        # x = x
        
        ### Keep or remove softmax
        # x = F.softmax(x, dim=1)
        ### Keep or remove softmax

        ## Extra needed from PolarNet
        x = x.permute(0,2,3,1)
        new_shape = list(x.size())[:3] + [32,19]
        x = x.view(new_shape)
        x = x.permute(0,4,1,2,3)
        #################

        return x