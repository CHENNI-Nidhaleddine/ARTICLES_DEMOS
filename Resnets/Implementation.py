import torch
import torch.nn as nn

class block(nn.Module):
    # we have three inputs: 
    # in_channels: the number of input channels to that block (eg. 64 for Resnet50 first block)
    # inter_channels: the number of intermediate channels inside the block (eg. 64 for Resnet50 first block)
    # stride: which will be used for the second conv layer in the block
    def __init__(self,in_channels,inter_channels,stride):
        super().__init__()

        self.conv1=nn.Conv2d(in_channels,inter_channels,kernel_size=1,stride=1,padding=0,bias=False)

        self.conv2=nn.Conv2d(inter_channels,inter_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        # the third conv layer will has inter_channels*4 output channels
        self.conv3=nn.Conv2d(inter_channels,inter_channels*4,kernel_size=1,stride=1,padding=0,bias=False)

        self.relu=nn.ReLU() #activation function
        
        # batch normalization
        self.bn1=nn.BatchNorm2d(inter_channels)
        self.bn2=nn.BatchNorm2d(inter_channels)
        self.bn3=nn.BatchNorm2d(inter_channels*4)

        # downsampling used to augment the identity size so we can perform addition with the conv layer output
        self.identity_downsample = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        inter_channels * 4,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(inter_channels * 4),
                )

    def forward(self,x):
        identity=x.clone() # the identity
        # ---- first conv layer ----
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)

        # ---- second conv layer ----
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)

        # ---- third conv layer ----
        x=self.conv3(x)
        x=self.bn3(x)

        # Now add the identity to the block output ( and this is the shortcut)
        # We are using try except so that we use the downsampling only when necessary
        try:
            x=x+identity
        except:
            x=x+self.identity_downsample(identity)

        x=self.relu(x)
        
        return x


class Resnet(nn.Module):
    # we have five inputs
    # x2: How many blocks for conv2_x (eg. 3 in Resnet50)
    # x3: How many blocks for conv3_x (eg. 6 in Resnet50)
    # x4: How many blocks for conv4_x (eg. 4 in Resnet50)
    # x5: How many blocks for conv5_x (eg. 3 in Resnet50)
    # n_classes: number of classes that we have in our classification problem
    def __init__(self,x2,x3,x4,x5,n_classes=2):
        super().__init__()
        # The first conv layer with RGB image as input
        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False) #output 112*112

        # the first stack of blocks (conv2_x)
        layers2=[]
        layers2.append(block(64,64,1)) # first block has an input of 64
        # other block with input of 64*4
        for _ in range(x2-1):
            layers2.append(block(64*4,64,1))
        self.conv2_x=nn.Sequential(*layers2)

        # conv3_x
        layers3=[]
        layers3.append(block(256,128,2))
        for _ in range(x3-1):
            layers3.append(block(128*4,128,1))
        self.conv3_x=nn.Sequential(*layers3)
 
        # conv4_x
        layers4=[]
        layers4.append(block(512,256,2))
        for _ in range(x4-1):
            layers4.append(block(256*4,256,1))
        self.conv4_x=nn.Sequential(*layers4)

        # conv5_x
        layers5=[]
        layers5.append(block(1024,512,2))
        for _ in range(x5-1):
            layers5.append(block(512*4,512,1))
        self.conv5_x=nn.Sequential(*layers5)
        
        # final dense layer 
        self.dense=nn.Linear(512*4*7*7,n_classes)

        

    def forward(self,x):
        x=self.conv1(x)
        x=nn.BatchNorm2d(64)(x)
        x=nn.ReLU()(x)
        print(x.shape,"conv1")
        x=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)(x)
        x=self.conv2_x(x)
        print(x.shape,"conv2")
        x=self.conv3_x(x)
        print(x.shape,"conv3")
        x=self.conv4_x(x)
        print(x.shape,"conv4")
        x=self.conv5_x(x)
        print(x.shape,"conv5")
        x=nn.AvgPool2d(1,1)(x)
        x = x.reshape(x.shape[0], -1)
        x=self.dense(x)
        return x

