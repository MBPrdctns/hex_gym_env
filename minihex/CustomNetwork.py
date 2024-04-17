import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.distributions import CategoricalDistribution
#from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        self.extracted_features = ResNetExtractor()
    
        with th.no_grad():
            n_flatten = self.extracted_features(
                th.as_tensor(observation_space.sample()[None]).float().unsqueeze(0)
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, obs):
        features = self.extracted_features(obs.unsqueeze(0))
        features = features.view(features.size(0), -1)  # Flatten the tensor
        return self.linear(features)


class ResNetExtractor(nn.Module):
    def __init__(self):
        super(ResNetExtractor, self).__init__()
        self.conv1 = convolutional(1, 32, 4) 
        self.residual1 = residual(32, 32, 4)
        self.residual2 = residual(32, 32, 4)
        self.residual3 = residual(32, 32, 4)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        x = self.flatten(x)
        return x

def convolutional(in_channels, filters, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, filters, kernel_size=kernel_size, stride=1, padding='same'),
        nn.BatchNorm2d(filters),
        nn.ReLU()
    )


def residual(in_channels, filters, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, filters, kernel_size=kernel_size, stride=1, padding='same'),
        nn.BatchNorm2d(filters),
        nn.ReLU(),
        nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=1, padding='same'),
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