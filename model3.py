
import torch.nn as nn
import torch


class FAC_CNN(nn.Module):
    def __init__(self, model):
        super(FAC_CNN, self).__init__()
        layer = list(model.features)

        self.conv1_1 = layer[0]
        self.relu1_1 = layer[1]
        self.conv1_2 = layer[2]
        self.relu1_2 = layer[3]
        self.maxpool1 = layer[4]

        self.conv2_1 = layer[5]
        self.relu2_1 = layer[6]
        self.conv2_2 = layer[7]
        self.relu2_2 = layer[8]
        self.maxpool2 = layer[9]

        self.conv3_1 = layer[10]
        self.relu3_1 = layer[11]
        self.conv3_2 = layer[12]
        self.relu3_2 = layer[13]
        self.conv3_3 = layer[14]
        self.relu3_3 = layer[15]
        self.maxpool3 = layer[16]

        self.conv4_1 = layer[17]
        self.relu4_1 = layer[18]
        self.conv4_2 = layer[19]
        self.relu4_2 = layer[20]
        self.conv4_3 = layer[21]
        self.relu4_3 = layer[22]
        self.maxpool4 = layer[23]

        self.conv5_1 = layer[24]
        self.relu5_1 = layer[25]
        self.conv5_2 = layer[26]
        self.relu5_2 = layer[27]
        self.conv5_3 = layer[28]
        self.relu5_3 = layer[29]
        self.maxpool5 = layer[30]

        # 高级语义特征提取层
        self.maxpool6 = nn.MaxPool2d((2, 2), stride=2)
        self.maxpool7 = nn.MaxPool2d((2, 2), stride=2)
        self.conv1x1_1 = nn.Conv2d(256, 1024, (1, 1), stride=(1, 1))
        self.conv1x1_2 = nn.Conv2d(512, 1024, (1, 1), stride=(1, 1))
        self.avg_pool2 = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        #         model.classifier._modules['6'] = nn.Linear(4096, 30)
        #         self.fc_layer = nn.Sequential(*list(model.children())[-1:])

        self.fc1 = model.classifier._modules['0']
        self.fc2 = model.classifier._modules['1']
        self.fc3 = model.classifier._modules['2']
        model.classifier._modules['4'] = nn.Linear(4096, 3072)
        self.fc4 = model.classifier._modules['3']
        self.fc5 = model.classifier._modules['4']
        self.fc6 = model.classifier._modules['5']
        model.classifier._modules['6'] = nn.Linear(4096, 30)
        self.fc7 = model.classifier._modules['6']

    def forward(self, x):
        outputs = []
        x = self.conv1_1(x)
        outputs.append(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        outputs.append(x)
        x = self.relu1_2(x)
        x = self.maxpool1(x)

        x = self.conv2_1(x)
        outputs.append(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        outputs.append(x)
        x = self.relu2_2(x)
        x = self.maxpool2(x)

        x = self.conv3_1(x)
        outputs.append(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        outputs.append(x)
        x = self.relu3_2(x)
        x1 = self.conv3_3(x)  # 56*56*256
        outputs.append(x)
        x = self.relu3_3(x1)
        x = self.maxpool3(x)

        x = self.conv4_1(x)
        outputs.append(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        outputs.append(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        outputs.append(x)
        x = self.relu4_3(x)
        x = self.maxpool4(x)

        x2 = self.conv5_1(x)  # 14*14*512
        outputs.append(x)
        x = self.relu5_1(x2)
        x = self.conv5_2(x)
        outputs.append(x)
        x = self.relu5_2(x)
        x3 = self.conv5_3(x)  # 14*14*256
        outputs.append(x3)
        x = self.relu5_3(x3)
        x = self.maxpool5(x)

        x = self.avg_pool(x)
        x = x.view(-1, 512 * 7 * 7)

        # 高级语义特征提取
        x1 = self.maxpool6(x1)
        x1 = self.maxpool7(x1)
        x1 = self.conv1x1_1(x1)
        x2 = self.conv1x1_2(x2)
        x3 = self.conv1x1_2(x3)

        res = torch.add(x1, x2)
        outputs.append(res)
        res = torch.add(res, x3)
        outputs.append(res)
        res = self.avg_pool2(res)
        res = res.view(-1, 1024 * 1 * 1)
        #         print(res.size())

        #         x = self.fc_layer(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = torch.cat((x, res), dim=1)
        x = self.fc6(x)
        x = self.fc7(x)

        return outputs
