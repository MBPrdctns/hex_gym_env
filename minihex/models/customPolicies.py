import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

class CustomPolicy(MaskableActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=torch.nn.Tanh,
                 *args, device='auto', **kwargs):
        super(CustomPolicy, self).__init__(observation_space, action_space,
                                          lr_schedule, net_arch, activation_fn,
                                          *args, **kwargs)

        # self.device = device

        extracted_features = self.resnet_extractor(self.processed_obs, **kwargs)
        self._policy = self.policy_head(extracted_features)
        self._value_fn = self.value_head(extracted_features)[0]
        self.q_value = self.value_head(extracted_features)[1]
        

        self._proba_distribution  = torch.distributions.Categorical(self._policy)

    def forward(self, obs, deterministic=False):
        
        if deterministic:
            actions, value, neglogp = self.predict(obs.to(self.device))
        else:  
            actions, value, neglogp = self.predict(obs.to(self.device))
        return actions, value, neglogp

    def proba_step(self, obs):
        return self.predict(obs, deterministic=False)

    def value(self, obs):
        return self.predict(obs)

    def value_head(self, y):
        y = self.convolutional(y, 4, 1)
        y = nn.Flatten()(y)
        y = self.dense(y, 128)
        vf = self.dense(y, 1, activation_fn=nn.Tanh())
        q = self.dense(y, 7, activation_fn=nn.Tanh())
        return vf, q

    def policy_head(self, y):
        y = self.convolutional(y, 4, 1)
        y = nn.Flatten()(y)
        policy = self.dense(y, 7)
        return policy

    def resnet_extractor(self, y, **kwargs):
        y = self.convolutional(y, 32, 4)
        y = self.residual(y, 32, 4)
        y = self.residual(y, 32, 4)
        y = self.residual(y, 32, 4)
        return y

    def convolutional(self, y, filters, kernel_size):
        y = nn.Conv2d(filters, kernel_size=kernel_size, stride=1, padding='same')(y)
        y = nn.BatchNorm2d(filters)(y)
        y = F.relu(y)
        return y

    def residual(self, y, filters, kernel_size):
        shortcut = y

        y = nn.Conv2d(filters, kernel_size=kernel_size, stride=1, padding='same')(y)
        y = nn.BatchNorm2d(filters)(y)
        y = F.relu(y)

        y = nn.Conv2d(filters, kernel_size=kernel_size, stride=1, padding='same')(y)
        y = nn.BatchNorm2d(filters)(y)
        y = F.relu(y + shortcut)

        return y

    def dense(self, y, filters, batch_norm=True, activation_fn=F.relu):
        y = nn.Linear(y, filters)

        if batch_norm:
            y = nn.BatchNorm1d(filters)(y)

        if activation_fn:
            y = activation_fn(y)

        return y
