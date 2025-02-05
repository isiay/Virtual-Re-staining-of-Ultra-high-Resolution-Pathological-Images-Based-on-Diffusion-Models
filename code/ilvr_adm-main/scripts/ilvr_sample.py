import argparse
from collections import OrderedDict
import datetime
import os
# import transformers
# from transformers import TrainingArguments
import sys   #导入sys模块
sys.path.append("/ilvr_adm-main")

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.image_datasets import load_data
from torchvision import utils
from resizer import Resizer
import math


# added
def load_reference(data_dir, batch_size, image_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        deterministic=True,
        random_flip=False,
    )
    for large_batch, model_kwargs,name in data:
        model_kwargs["ref_img"] = [large_batch,name]
        yield model_kwargs


def main():
    args = create_argparser().parse_args()

    # th.manual_seed(0)

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)
    logger.log(args)
    logger.log("creating model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )


    # model.load_state_dict(
    #     dist_util.load_state_dict(args.model_path, map_location="cpu")
    # )

    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")  # 模型pth文件
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # 去掉前缀（去掉前七个字符）
        new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
    # net.load_state_dict(new_state_dict, strict=True)  # 重新加载这个模型
    model.load_state_dict(
            new_state_dict
        )
    
    model = th.nn.DataParallel(model)
   

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating resizers...")
    print(args.down_N)
    print(math.log(args.down_N, 2))
    assert math.log(args.down_N, 2).is_integer()

    shape = (args.batch_size, 3, args.image_size, args.image_size)
    shape_d = (args.batch_size, 3, int(args.image_size / args.down_N), int(args.image_size / args.down_N))
    down = Resizer(shape, 1 / args.down_N).to(next(model.parameters()).device)
    up = Resizer(shape_d, args.down_N).to(next(model.parameters()).device)
    resizers = (down, up)

    logger.log("loading data...")
    data = load_reference(
        args.base_samples,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("creating samples...")
    count = 0
    while count * args.batch_size < args.num_samples:
    # while count < len():
        model_kwargs_old = next(data)
        
        model_kwargs = {}
        for k, v in model_kwargs_old.items():
            model_kwargs[k] = v[0].to(dist_util.dev())
            names=v[1]


        # model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            resizers=resizers,
            range_t=args.range_t
        )

        for i in range(args.batch_size):
            out_path = os.path.join(logger.get_dir(),
                                    names[i]+".png")
            utils.save_image(
                sample[i].unsqueeze(0),
                out_path,
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

        count += 1
        logger.log(f"created {count * args.batch_size} samples")
        sys.stdout.write('\rtrain batch %04d of %04d' % (count * args.batch_size, args.num_samples))

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    N=2#1
    t=250#150
    defaults = dict(
        clip_denoised=True,
        num_samples=3276,
        batch_size=64,
        down_N=N,
        range_t=t,
        use_ddim=False,
        base_samples="/10X_data/256_noover/test/x",
        model_path="/guided-diffusion/logger/2024-03-05-11-49-22-111876/model007800.pt",
        save_dir="/ilvr/test/"+"model7800{}_t_{}_N".format(t,N)+"_"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"),
        save_latents=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()