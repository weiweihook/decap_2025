import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from typing import Optional, Tuple



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
    """A categorical distribution that supports invalid actions via boolean masks."""
    def __init__(self, probs=None, logits=None, validate_args=None, masks=None):
        # If no masks are supplied fall back to the default behaviour
        if masks is None:
            super().__init__(probs=probs, logits=logits, validate_args=validate_args)
            self.masks = None
        else:
            # Ensure mask is boolean and on the same device as logits
            self.masks = masks.to(dtype=torch.bool, device=logits.device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e8, device=logits.device))
            super().__init__(probs=probs, logits=logits, validate_args=validate_args)

    def entropy(self):
        # Delegate to parent implementation when no mask is provided
        if self.masks is None:
            return super().entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0, device=self.logits.device))
        return -p_log_p.sum(-1)

class Encoder(nn.Module):
    
    def __init__(self, input_channels: int, use_batch_norm: bool = True):
        super().__init__()
        
        self.use_batch_norm = use_batch_norm
        
        # 第一个卷积块
        self.conv1 = layer_init(nn.Conv2d(input_channels, 32, kernel_size=3, padding=1))
        self.bn1 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # 第二个卷积块
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1))
        self.bn2 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # 第三个卷积块
        self.conv3 = layer_init(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        self.bn3 = nn.BatchNorm2d(128) if use_batch_norm else nn.Identity()
        self.pool3 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # 第四个卷积块
        self.conv4 = layer_init(nn.Conv2d(128, 256, kernel_size=3, padding=1))
        self.bn4 = nn.BatchNorm2d(256) if use_batch_norm else nn.Identity()
        
        # 使用更稳定的激活函数
        self.activation = nn.ReLU(inplace=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 第一层：11x11 -> 6x6
        x = self.pool1(self.activation(self.bn1(self.conv1(x))))
        
        # 第二层：6x6 -> 3x3
        x = self.pool2(self.activation(self.bn2(self.conv2(x))))
        
        # 第三层：3x3 -> 2x2
        x = self.pool3(self.activation(self.bn3(self.conv3(x))))
        
        # 第四层
        x = self.activation(self.bn4(self.conv4(x)))
        
        return x  # 输出: [batch, 256, 2, 2]

class Decoder(nn.Module):
    
    def __init__(self, output_channels: int, use_batch_norm: bool = True):
        super().__init__()
        
        self.use_batch_norm = use_batch_norm
        
        # 上采样层1: 2x2 -> 4x4
        self.deconv1 = layer_init(nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.bn1 = nn.BatchNorm2d(256) if use_batch_norm else nn.Identity()
        
        # 上采样层2: 4x4 -> 8x8
        self.deconv2 = layer_init(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.bn2 = nn.BatchNorm2d(128) if use_batch_norm else nn.Identity()
        
        # 上采样层3: 8x8 -> 11x11
        self.deconv3 = layer_init(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1, padding=0))
        self.bn3 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        
        # 精细化层
        self.conv1 = layer_init(nn.Conv2d(64, 32, kernel_size=3, padding=1))
        self.bn4 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        
        # 输出层
        self.conv_out = layer_init(nn.Conv2d(32, output_channels, kernel_size=3, padding=1))
        
        # 维度转换
        self.transpose = Transpose((0, 2, 3, 1))  # NCHW -> NHWC
        
        # 激活函数
        self.activation = nn.ReLU(inplace=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 上采样序列
        x = self.activation(self.bn1(self.deconv1(x)))  # 2x2 -> 4x4
        x = self.activation(self.bn2(self.deconv2(x)))  # 4x4 -> 8x8
        x = self.activation(self.bn3(self.deconv3(x)))  # 8x8 -> 11x11
        
        # 精细化
        x = self.activation(self.bn4(self.conv1(x)))
        
        # 输出层
        x = self.conv_out(x)
        
        # 维度转换: [batch, C, 11, 11] -> [batch, 11, 11, C]
        return self.transpose(x)


class PPONetwork(nn.Module):
    """
    优化后的PPO网络
    改进了内存使用、计算效率和数值稳定性
    """
    
    def __init__(self, 
                 env, 
                 input_channels: int = 5, 
                 use_batch_norm: bool = False,
                 dropout_rate: float = 0.0):
        super().__init__()
        
        self.action_space = env.ACTION_SPACE
        self.input_channels = input_channels
        self.use_batch_norm = use_batch_norm
        
        # 阻抗处理网络（添加dropout以防止过拟合）
        self.imped_fc = nn.Sequential(
            layer_init(nn.Linear(4 * 231, 512), std=1),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            layer_init(nn.Linear(512, 2 * 121), std=1)
        )
        
        # 编码器（输入通道数包括原始输入和阻抗特征）
        self.encoder = Encoder(input_channels + 2, use_batch_norm)
        
        # Actor网络（策略网络）
        self.actor = Decoder(len(env.ACTION_MEANINGS), use_batch_norm)
        
        # Critic网络（价值网络)
        self.critic = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(256 * 2 * 2, 512), std=1),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            layer_init(nn.Linear(512, 256), std=1),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            layer_init(nn.Linear(256, 1), std=1),
        )
    
    def _process_impedance(self, imped: torch.Tensor) -> torch.Tensor:
        """处理阻抗输入"""
        batch_size = imped.shape[0]
        processed = self.imped_fc(imped)
        return processed.reshape(batch_size, 2, 11, 11)


    def _encode_features(self, x: torch.Tensor, imped: torch.Tensor) -> torch.Tensor:
        """
            编码特征
        """
        # 处理阻抗特征
        imped_features = self._process_impedance(imped)
        
        # 拼接输入特征
        combined_input = torch.cat((x, imped_features), dim=1)
        
        # 编码
        encoded_features = self.encoder(combined_input)
        
        return encoded_features

    def get_value(self, x: torch.Tensor, imped: torch.Tensor) -> torch.Tensor:
        """
        获取状态价值
        
        Args:
            x: 状态特征 [batch, channels, height, width]
            imped: 阻抗特征 [batch, 4*231]
            
        Returns:
            状态价值 [batch, 1]
        """
        encoded_features = self._encode_features(x, imped)
        return self.critic(encoded_features)

    def get_action_and_value(self, 
                           x: torch.Tensor, 
                           imped: torch.Tensor, 
                           action_mask: torch.Tensor, 
                           action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取动作和价值
        
        Args:
            x: 状态特征
            imped: 阻抗特征
            action_mask: 动作掩码
            action: 可选的指定动作
            
        Returns:
            (动作, 对数概率, 熵, 价值)
        """
        # 编码特征
        encoded_features = self._encode_features(x, imped)
        
        # 获取动作logits
        logits = self.actor(encoded_features)
        logits = logits.reshape(encoded_features.shape[0], -1)
        
        # 分割logits和掩码
        split_logits = torch.split(logits, self.action_space.tolist(), dim=1)
        split_action_masks = torch.split(action_mask, self.action_space.tolist(), dim=1)
        
        # 创建掩码分类分布
        multi_categoricals = [
            CategoricalMasked(logits=logits_i, masks=mask_i) 
            for logits_i, mask_i in zip(split_logits, split_action_masks)
        ]
        
        # 采样或使用指定动作
        if action is None:
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        
        # 计算对数概率和熵
        logprob = torch.stack([
            categorical.log_prob(a) 
            for a, categorical in zip(action, multi_categoricals)
        ])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        
        # 获取价值
        value = self.critic(encoded_features)
        
        return action.T, logprob.sum(0), entropy.sum(0), value
    

