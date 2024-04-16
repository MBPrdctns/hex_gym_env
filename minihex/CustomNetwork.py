import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.distributions import CategoricalDistribution

class CustomPolicy(nn.Module):
    def __init__(self, ob_space_shape, ac_space_shape, **kwargs):
        super(CustomPolicy, self).__init__(ob_space_shape, ac_space_shape, **kwargs)

        self.extracted_features = ResNetExtractor(ob_space_shape, **kwargs)
        self.policy_head = PolicyHead(self.extracted_features)
        self.value_head = ValueHead(self.extracted_features)

        self.proba_distribution = CategoricalDistribution()

    def forward(self, obs):
        extracted_features = self.extracted_features(obs)
        policy = self.policy_head(extracted_features)
        value = self.value_head(extracted_features)
        return policy, value

    def step(self, obs, deterministic=False):
        with torch.no_grad():
            policy, value = self.forward(obs)
            if deterministic:
                action = policy.argmax(dim=-1, keepdim=True)
            else:
                action = self.proba_distribution.sample(policy)
            return action, value, None, None

    def value(self, obs):
        with torch.no_grad():
            _, value = self.forward(obs)
            return value


class ResNetExtractor(nn.Module):
    def __init__(self, input_shape, **kwargs):
        super(ResNetExtractor, self).__init__()
        self.conv1 = convolutional(1, 32, 4)  # Adjusted for single channel input
        self.residual1 = residual(32, 32, 4)
        self.residual2 = residual(32, 32, 4)
        self.residual3 = residual(32, 32, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        return x


class PolicyHead(nn.Module):
    def __init__(self, input_shape):
        super(PolicyHead, self).__init__()
        self.fc = dense(input_shape[0] * input_shape[1] * input_shape[2], 7, batch_norm=False, activation=None)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ValueHead(nn.Module):
    def __init__(self, input_shape):
        super(ValueHead, self).__init__()
        self.conv = convolutional(input_shape[0], 4, 1)
        self.flatten = nn.Flatten()
        self.fc1 = dense(4 * input_shape[1] * input_shape[2], 128, batch_norm=False)
        self.fc_vf = dense(128, 1, batch_norm=False, activation='tanh')
        self.fc_q = dense(128, 7, batch_norm=False, activation='tanh')

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        vf = self.fc_vf(x)
        q = self.fc_q(x)
        return vf, q


def convolutional(in_channels, filters, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, filters, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2),
        nn.BatchNorm2d(filters),
        nn.ReLU()
    )


def residual(in_channels, filters, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, filters, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2),
        nn.BatchNorm2d(filters),
        nn.ReLU(),
        nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2),
        nn.BatchNorm2d(filters),
        nn.ReLU()
    )


def dense(in_features, out_features, batch_norm=True, activation='relu'):
    layers = []
    layers.append(nn.Linear(in_features, out_features))
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_features))
    if activation:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)