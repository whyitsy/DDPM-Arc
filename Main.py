from Diffusion.Train import train, eval

# 使用ipex，1.8it/s,
# 使用4080，5.7it/s

"""问题：
参考文章之一: https://juejin.cn/post/7258069406961352764

这里使用dropout是有什么原因吗,现在一般不都是使用BN吗?
grad_clip这里是第一次遇见,是新技术吗(可能是最近几年的新技术,而我并没有)

回答: BN并没有取代之前的技术,应该只是多数情况下,BN之后就很稳定了,不需要其他技术了。但是在特定的场景还是需要使用其他方法的
"""


def main(model_config = None):
    modelConfig = {
        "state": "eval", # train or eval
        "epoch": 200,
        "batch_size": 80,
        "T": 1000,  # 用于扩散的次数上限，max
        "channel": 128, # basechannel, Unet每次下采样后的channel数都是上一次的倍数, 所以用一个basechannel方便表示
        "channel_mult": [1, 2, 3, 4], # Unet每次下采样后的channel数, 用basechannel的倍数表示
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15, 
        "lr": 1e-4, 
        "multiplier": 2.,# 预热学习率的放大倍数
        "beta_1": 1e-4,  # x和noise的系数 β min, 该系数是自定义的
        "beta_T": 0.02,  # 系数 β max
        "img_size": 32,  # 图片的长宽大小
        "grad_clip": 1., # 梯度裁剪（Gradient Clipping）,通过限制模型参数梯度更新的最大值(默认是L2范数)，防止梯度爆炸。
        ############ change ############
        # "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        "device":"xpu",
        ############ change ############
        # "training_load_weight": None,
        "training_load_weight": "ckpt_136_.pt",
        "save_weight_dir": "./Checkpoints/",
        # "test_load_weight": "ckpt_199_.pt",
        "test_load_weight": "ckpt_136_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8  # 每一行显示的图片数量 
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
