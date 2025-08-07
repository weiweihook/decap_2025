import numpy as np
import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
# device = 'cpu'
EPS = 1e-5

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Transpose(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(device))
        return -p_log_p.sum(-1)

class Encoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),  # 11x11 -> 6x6
            nn.ReLU(),
            
            layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),  # 6x6 -> 3x3
            nn.ReLU(),
            
            layer_init(nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),  # 3x3 -> 2x2
            nn.ReLU(),
            
            layer_init(nn.Conv2d(128, 256, kernel_size=3, padding=1)),
        )
    
    def forward(self, x):
        return self.encoder(x)  # output: [b,256,2,2]

class Decoder(nn.Module):
    def __init__(self, output_channels):
        super().__init__()

        self.decoder = nn.Sequential(
            # 2x2 -> 4x4
            layer_init(nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),

            # 4x4 -> 8x8
            layer_init(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),

            # 8x8 -> 11x11
            layer_init(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1, padding=0)),
            nn.ReLU(),

            layer_init(nn.Conv2d(64, 32, kernel_size=3, padding=1)),
            nn.ReLU(),

            # output layer
            layer_init(nn.Conv2d(32, output_channels, kernel_size=3, padding=1)),
            Transpose((0, 2, 3, 1)),  # NCHW -> NHWC: [b,C,11,11] -> [b,11,11,C]
        )

    def forward(self, x):
        return self.decoder(x)


class PPONetwork(nn.Module):
    def __init__(self, env, input_channels=5):
        super(PPONetwork, self).__init__()
        self.action_space = env.action_space
        self.imped_fc = nn.Sequential(layer_init(nn.Linear(4*231, 2*121), std=1))
        self.encoder = Encoder(input_channels+2)
        self.actor = Decoder(len(env.action_meaning)) #200pF
        self.critic = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(256 * 2 * 2, 512), std=1),
            nn.ReLU(),
            layer_init(nn.Linear(512, 256), std=1),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1),
            )

    def get_value(self, x, imped):
        i = self.imped_fc(imped).reshape(imped.shape[0], 2, 11, 11)
        xi = torch.cat((x, i), dim=1)
        x0 = self.encoder(xi)

        return self.critic(x0)

    def get_action_and_value(self, x, imped, action_mask, action=None):
        i = self.imped_fc(imped).reshape(imped.shape[0], 2, 11, 11)
        xi = torch.cat((x, i), dim=1)
        x0 = self.encoder(xi)
        logits = self.actor(x0).reshape(x0.shape[0], -1)
        split_logits = torch.split(logits, self.action_space.tolist(), dim=1)
        split_action_masks = torch.split(action_mask, self.action_space.tolist(), dim=1)
        multi_categoricals = [
            CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_action_masks)
        ]
        if action is None:
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action.T, logprob.sum(0), entropy.sum(0), self.critic(x0)

