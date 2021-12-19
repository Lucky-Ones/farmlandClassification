import torch.nn as nn


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

        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        model.classifier._modules['6'] = nn.Linear(4096, 4)
        self.fc_layer = nn.Sequential(*list(model.children())[-1:])

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.maxpool1(x)

        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.maxpool2(x)

        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        x = self.maxpool3(x)

        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        x = self.maxpool4(x)

        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.relu5_3(x)
        x = self.maxpool5(x)

        x = self.avg_pool(x)
        x = x.view(-1, 512 * 7 * 7)
        #         print(x.size())

        x = self.fc_layer(x)
        return x
