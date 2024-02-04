import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=13, condition_dim=3):
        super(Generator, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.condition_dim = condition_dim
        self.input_dim = input_dim + condition_dim
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Unflatten(1, (16, 4, 4)),  # Adjust the numbers to ensure correct reshaping
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, output_dim, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Normalize the output to the range [-1, 1]
        ).to(device)

    def forward(self, noise, conditions):
        gen_input = torch.cat((noise, conditions), -1)
        return self.model(gen_input)


class Discriminator(nn.Module):
    def __init__(self, input_dim=13, condition_dim=3):
        super(Discriminator, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.condition_dim = condition_dim
        self.adv_layer = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ZeroPad2d((0, 1, 0, 1)),  # Padding to adjust for size during convolutions
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Flatten(),
        ).to(device)

        self._to_linear = None
        self.flatten_size = self._get_flatten_size(torch.zeros(1, input_dim, 8, 8, device=device))

        # Combining the flattened conv output with the condition
        self.decision_layer = nn.Sequential(
            nn.Linear(self.flatten_size + condition_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        ).to(device)

    def _get_flatten_size(self, x):
        if self._to_linear is None:
            self._to_linear = self.adv_layer(x).shape[1]
        return self._to_linear

    def forward(self, img, conditions):
        img_adv = self.adv_layer(img)
        combined = torch.cat((img_adv, conditions.view(conditions.size(0), -1)), dim=1)
        validity = self.decision_layer(combined)
        return validity
