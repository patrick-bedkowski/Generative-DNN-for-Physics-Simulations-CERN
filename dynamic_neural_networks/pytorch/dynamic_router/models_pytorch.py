import torch.nn as nn
import torch


# Define the Generator model
class Generator(nn.Module):
    def __init__(self, noise_dim, cond_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, 256),  # This should be 19 (10 + 9)
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128 * 20 * 10),
            nn.BatchNorm1d(128 * 20 * 10),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        self.upsample = nn.Upsample(scale_factor=(3, 2))
        self.conv_layers = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(2, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Upsample(scale_factor=(1, 2)),
            nn.Conv2d(256, 128, kernel_size=(2, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv2d(128, 64, kernel_size=(2, 2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv2d(64, 1, kernel_size=(2, 7)),
            nn.ReLU()
        )

    def forward(self, noise, cond):
        x = torch.cat((noise, cond), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 128, 20, 10)
        x = self.upsample(x)
        x = self.conv_layers(x)
        return x

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self, cond_dim, num_generators):
        super(Discriminator, self).__init__()
        self.num_generators = num_generators
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 16, kernel_size=(3, 3)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 12 * 12 + cond_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        self.fc3 = nn.Linear(64, self.num_generators)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, cond):
        x = self.conv_layers(img)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, cond), dim=1)
        x = self.fc1(x)
        latent = self.fc2(x)
        out = self.fc3(latent)
        out = self.sigmoid(out)
        return out, latent

# # Define the Router Network
# class RouterNetwork(nn.Module):
#     def __init__(self, cond_dim):
#         super(RouterNetwork, self).__init__()
#         self.fc_layers = nn.Sequential(
#             nn.Linear(cond_dim, 128),
#             nn.LeakyReLU(0.1),
#             nn.Linear(128, 64),
#             nn.LeakyReLU(0.1),
#             nn.Linear(64, 32),
#             nn.LeakyReLU(0.1),
#             nn.Linear(32, 3),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, cond):
#         return self.fc_layers(cond)


# Define the Router Network
class RouterNetwork(nn.Module):
    def __init__(self, cond_dim):
        super(RouterNetwork, self).__init__()
        self.fc_layers = nn.Sequential(
            # Hidden Layer 1
            nn.Linear(cond_dim, 128),
            nn.BatchNorm1d(128),  # Batch normalization
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),  # Dropout for regularization

            # Hidden Layer 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),  # Batch normalization
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),  # Dropout for regularization

            # Hidden Layer 3
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),  # Batch normalization
            nn.LeakyReLU(0.1),

            # Output Layer
            nn.Linear(32, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, cond):
        return self.fc_layers(cond)


class AuxReg(nn.Module):
    def __init__(self):
        super(AuxReg, self).__init__()

        self.conv3 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv3_bd = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4_bd = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1))

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv5_bd = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1)
        )
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 1))

        self.conv6 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv6_bd = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1)
        )

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(256 * 3 * 8, 2)  # Adjust the input dimension based on your input image size

    def forward(self, x):
        x = self.conv3(x)
        x = self.conv3_bd(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.conv4_bd(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.conv5_bd(x)
        x = self.pool5(x)

        x = self.conv6(x)
        x = self.conv6_bd(x)

        x = self.flatten(x)
        x = self.dense(x)

        return x

