import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    # 在dim=0上获取v中index=t位置的值
    out = torch.gather(v, index=t, dim=0).float().to(device)
    # out.view将其形状改为[batch_size,1,1,1],这里的三个1维度是为了对应后面的channel，height，width。
    # 在进行广播计算后，就能够将这一个系数应用于所有的采样点上了
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        # Unet
        self.model = model
        # 最大迭代轮数
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).float())
        alphas = 1. - self.betas # 一维
        # 累乘
        alphas_bar = torch.cumprod(alphas, dim=0) # 一维
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        不管t是多少, 都是根据推导的alphas_bar一步到位, 加噪和预测噪声都是
        """
        # 生成batch_size个[0,T-1]之间的随机整数。
        # t为随机生成迭代轮数的列表
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        # reduction表示如何对损失进行聚合，如返回平均、求和、不处理(返回每个值)
        # 在高斯分布中随机采样的噪声就是ground truth, 我们要预测的都是高斯分布采样的随机高斯噪声。
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss

# 这个是推理时使用的
class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).float())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        
        # [1,0]表示在最后一个维度的前端添加一个值，后端添加零个值(就是不加)，value=1为添加的值
        # alphas_bar是一维tensor, 所以这里就相当于在alphas_bar的前面添加了一个1,然后取前T个，用于表示前向扩散时X_t的系数(分布方差)。
        # 这里的操作说明前向和反向的过程不是对称的，前向是每一步都有添加噪音(噪音系数sqrt(1-alphas_bar)!=0)
        # 而反向为了最后一步要完全去掉噪声生成图像，所以最后一步噪音的系数为0,即不加噪音，体现论文公式的 z~N(0,1)if t>1,else z=0
        # 对应到SD中经常出现最后一步突然从模糊图像变为清晰图像就是去噪step不够，而又要在最后一步完全去噪导致的
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # 对应论文公式第四行，将外面的括号打开，系数乘进去了
        self.register_buffer('coeff1', torch.sqrt(1. / alphas)) 
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        # 用于计算方差σ_t的, 不知道为什么这么写
        # (1. - alphas_bar_prev)是小于(1. - alphas_bar)一个错位的
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    # 计算均值, 不知道为什么这个叫做均值
    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        # Sampling算法的第四行右边第一部分
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    # 计算均值和方差。没看推导不知道为什么这么写, 就只是按照公式理解。
    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        # Sampling算法的第四行右边第一部分
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            # x_t.new_ones:在x_t上使用new_ones的作用是生成的数据和x_t在相同的设备上(如CPU或不同的GPU)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            # 均值和方差
            mean, var= self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            # Sampling算法的第四行右边第二部分 σ_t*z
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   


