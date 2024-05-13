import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class Swish(nn.Module):
    """
    Swish激活函数, Google在17年提出。一般优于传统激活函数, 如Relu
    这时是无参数的版本, 还有一个存在一个参数的Swish-B。
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    TimeEmbedding模块将把整型t, 以Transformer函数式位置编码的方式, 映射成向量
    其shape为(batch_size, time_channel)
    """
    def __init__(self, T, d_model, dim):
        """
        T: 扩散的次数, 这里是建立好max_step的map, 后面的随机step就可以直接获取对应step的temb
        d_model: embedding的维度
        """
        assert d_model % 2 == 0
        super().__init__()

        # embedding编码, 同Transformer的Positional encoding
        ##### i/{ 10000^{2*j/d} } ###############
        # 这里只计算了一半, 另一半通过stack合并。 计算应该也能够使用pow
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000) 
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        # 这里通过广播建立一个所有step的temb的map, 一次Unet一个temb
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        #########################################

        # stack用于将多个tensor合并为一个tensor, 按照dim扩展新的维度
        # emb只包含了一半, 使用stack合并每一个emb的sin、cos就能得到每两个编码一组的列表, 奇数和偶数
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]

        # 最后将列表打开合并, 结果就是相邻的一组奇数和偶数的sin、cos的参数相同
        emb = emb.view(T, d_model)


        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        # 下采样不改变channel, 只是将长宽减半
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x

class UpSample(nn.Module):
    """
    上采样使用的最近邻插值加倍图像尺寸, 然后再卷积。与下采样对应相加之后就可以再接上采样了。
    """
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        # 插值算法, nearest: 最近邻插值算法
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        # 自注意力
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        # 调整维度, 用以矩阵乘法
        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        # 矩阵乘法,  (int(C) ** (-0.5))是后面沿着channel做Attention时的缩放
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        # 这个概率就是注意力分数了吧！
        w = F.softmax(w, dim=-1) 


        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        # additive attention
        return x + h


class ResBlock(nn.Module):
    """
    在初版Unet中, 这里就只是ResBlock中的block1。而这里需要加入temb和使用Attention, 并且Encoder和decoder都需要使用。
    这个结构是参考 https://juejin.cn/post/7251391372394053691  , 具体是哪里提出来的不知道
    """
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()

        # block1是原版变化, 改变channel
        # 然后temb_proj对temb进行投影, 对齐channel数, 为[batch,ch]维度。投影后还需要扩展两个维度然后使用广播和图片[batch_size, out_ch, H, W]相加
        # 相加之后在进行一次卷积, 对应block2
        # 然后残差连接原图x
        # 最后进行Attention, 如果有的话

        ## 但是为什么是这个结构呢？

        self.block1 = nn.Sequential(
            # GroupNorm是BN的一种泛化。输入通道被分成多个组，每组内的通道使用相同的归一化参数进行归一化。
            # GroupNorm特别适用于小批量大小的情况，Batch Normalization 可能无法有效地工作，因为批量统计信息的估计不够准确。
            # 注意in_ch需要是32的倍数
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )

        # temb投影到out_ch的维度。相同维度才能直接相加
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )

        # 1x1卷积调整输入和输出通道不同
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        # 如果相同则直接通过, 不作操作
        else:
            self.shortcut = nn.Identity()

        # 如果包含注意力机制，则添加AttnBlock
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    # 初始化权重和偏置
    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        # 后面[:, :, None, None]是对temb_proj的结果进行reshape, 后面的None就是占位用的, 表示获取所有一维和二维数据, 在扩展三维和四维=>[batch_size, out_ch, 1, 1] 
        # 这样后就可以广播为[batch_size, out_ch, H, W]的tensor. 这是常用的扩展维度的方法
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        # 残差连接
        h = h + self.shortcut(x)

        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        # 有padding的改进Unet, 不改变图片的尺寸
        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()

        # 记录下采样时输出的channel数, 用于上采样时进行拼接
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        # 搭建整体网络架构
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                # 下采样
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)

            # 不是最后一个
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
    print(y.shape)

