from Diffusion.Train import train, eval

# 使用ipex，1.8it/s,
# 使用4080，5.7it/s

def main(model_config = None):
    modelConfig = {
        "state": "eval", # or eval
        "epoch": 200,
        "batch_size": 80,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        ############ change ############
        # "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        "device":"xpu",
        ############ change ############
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        # "test_load_weight": "ckpt_199_.pt",
        "test_load_weight": "ckpt_1_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


# if __name__ == '__main__':
main()
