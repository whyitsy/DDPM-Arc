import os
from typing import Dict


import intel_extension_for_pytorch as ipex


import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image


from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler


"""
因为每一次的噪声都是采样得到的, 而就算是相同的分布每次采样都不一样, 故不同分布的采样就更不一样了。
所以只有记住每次加入的噪声, 才可以从x_t一步一步的减掉噪声还原x_0。

而从理论推导出发, 前向扩散相同, 可以记下每次加入的噪声; 
但是反向降噪是学习加入噪声的分布, 然后通过学习到的分布采样噪声, 再减掉噪声进行还原, 所以训练出来的网络是不可能百分百还原的, 只能很像原图。

我们训练的网络的目的是直接利用训练好的网络从一个高斯噪声中去噪生成需要的图片。单纯的加噪和降噪百分比还原是没有意义的。
就像1+100=101, 101-100=1一样。我们的目的是训练网络使其能够生成尽可能接近加入的100。
"""
def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    
    # pin_memory在多GPU环境中如果为true才会生效，在内存和GPU显存之间进行映射，这种映射不需要额外的CPU到GPU的数据传输。但会增加内存的使用，数据量大时可能会溢出，权衡使用。
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    # U-Net网络
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    

    # 加载权重
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    
    # 在Adam中引入了weight_decay
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    

    ########### ipex优化 ############ 
    # 可以使用dtype=torch.float16/32/64等，不过arc平台只能16或32精度
    net_model, optimizer = ipex.optimize(net_model, optimizer=optimizer)

    #######################
    
    
    # 余弦退火（Cosine Annealing）策略，它将学习率按照余弦函数的形式周期性地降低，以期达到更好的训练效果和更快的收敛。
    # T_max为Maximum number of iterations.在这个周期类，学习率下降到eta_min(default:0)。这里就是越到后面学习率越小
    # eta_min为学习率可以降到的最低值
    # last_epoch表示上一个训练的epoch，如果之前训练中途停止，可以输入已经训练的epoch数继续当时的学习率；-1表示之前没有训练过，从第一个epoch开始。该参数会在每次step()之后更新为下一个值(一般为+1)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=1e-6, last_epoch=-1)
    
    # 预热学习率，主要作用是在训练初期逐渐增加（warm-up）学习率，而不是从一开始就使用一个较高的学习率。预热可以减少训练初期由于较大学习率引起的梯度爆炸问题，还可以避免过早陷入局部最优解。
    # 但实际上预热学习率是一种经验方法，根据实践，在很多情况确实能够提供训练效率和性能。
    # 预热过程持续若干个epoch，在热身完毕之后可以衔接其他的学习率调度器，这里就是衔接cosineScheduler(CosineAnnealingLR)
    # multiplier参数为方法倍数，表示在热身完成之后的学习率是原来学习率的multiplier倍，根据经验设置的
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        # dynamic_ncols=True，则进度条的宽度会随着窗口的大小动态适应
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                # 这里除1000而不是batch_size应该只是经验，因为loss不能太大不然导致不稳定，而batch_size=80太小，所以就1000
                loss = trainer(x_0).sum() / 1000.
                loss.backward()
                # 防止梯度爆炸的技术
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                # 动态更新每个batch后的数据
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })

        warmUpScheduler.step()
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))


def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, 32, 32], device=device)
        # 高斯分布在三个标准差范围内[-3,3]的概率为99.7%。经过noisyImage * 0.5 + 0.5之后为N(0.5,0.25),绝大多数在[-0.25,1.25]
        # clamp函数(input,min,max),将[-0.25,1.25]内的值限制在了[0,1]
        # 而为什么要限制在[0,1]? 一般激活函数（如ReLU）和梯度下降算法效果会更好，减少梯度爆炸的风险。图像数据通常需要在 [0, 1] 范围内才能正确显示。
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        # 保存随机采样的高斯噪声图片
        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        
        # sampler返回生成的图像, 像素值被缩放到[-1,1]
        sampledImgs = sampler(noisyImage)
        # 缩放到[0,1], 便于保存
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(sampledImgs, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])