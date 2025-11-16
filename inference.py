import os
import torch
from aligngen.aligned_pipeline import FluxPipeline
from aligngen.attention_processor import ShareAttnFluxAttnProcessor2_0, StyleAlignedArgs
from aligngen.aligned_transformer import FluxTransformer2DModel
from PIL import Image
import torch
import numpy as np
import argparse
import json

# Flux下载的目录
cache_dir = "./Tdir"

# 初始化注意力处理器，将共享注意力模块应用到 Transformer 的所有注意力层
def init_attention_processors(pipeline: FluxPipeline, style_aligned_args: StyleAlignedArgs | None = None):
    attn_procs = {}
    transformer = pipeline.transformer
    # 遍历所有注意力处理器名称，逐层注入共享注意力机制
    for i, name in enumerate(transformer.attn_processors.keys()):
        attn_procs[name] = ShareAttnFluxAttnProcessor2_0(
            cnt=i,
            style_aligned_args=style_aligned_args,
        )
    transformer.set_attn_processor(attn_procs)

# 将多张图片按 k 列拼接成总图
def concat_img(images, k):
    img_width, img_height = images[0].size
    n = len(images)
    rows = (n + k - 1) // k

    total_width = img_width * k
    total_height = img_height * rows
    new_img = Image.new('RGB', (total_width, total_height), 'white')

    for i, img in enumerate(images):
        row = i // k
        col = i % k
        new_img.paste(img, (col * img_width, row * img_height))

    return new_img


def main(args):
    # 配置 Style-Aligned 参数
    # 设置共享注意力、作用的 block 范围、时间步范围以及风格控制强度
    style_args = StyleAlignedArgs(
        share_attention=True,
        block=(19, 57),            # 指定共享注意力应用的 Transformer Block 区间
        timesteps=(0, 30),         # 指定共享注意力的时间步范围
        style_lambda_mode="fix",
        style_lambda=args.style_lambda,  # 使用命令行传入的 style_lambda
        constrain_first=True
    )

    # 加载 FLUX Transformer 模型
    print(f"Loading model from: {args.model_path}")
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir
    )
    transformer.set_style_aligned_args(style_args)  # 注入风格对齐参数

    # 加载 Pipeline（图像生成主入口）
    pipe = FluxPipeline.from_pretrained(
        args.model_path,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir
    )

    # 初始化注意力共享模块
    init_attention_processors(pipe, style_args)

    # 启用 CPU 卸载以节省显存（显卡不足时非常重要）
    pipe.enable_model_cpu_offload()

    # 执行推理
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print("Generating images...")
    images = pipe(
        prompts,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=30,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images

    torch.cuda.empty_cache()

    # 保存生成的图片，并构建 prompt → image_name 的映射
    prompt_image_map = {}

    for i, image in enumerate(images):
        image_name = f"{i}.jpg"
        image_path = os.path.join(output_dir, image_name)
        image.save(image_path)
        prompt_image_map[prompts[i]] = image_name  # 建立 prompt 对应文件名的映射

    # 生成拼接图
    concat_result = concat_img(images, len(images))
    concat_result.save(f"{output_dir}/concat.jpg")

    json_str = json.dumps(prompt_image_map, ensure_ascii=False)
    print("Prompt → Image 映射：", json_str)
    json_path = os.path.join(output_dir, "prompt_image_map.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_str)

    print(f"Results saved to '{output_dir}' directory.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlignedGen Inference")

    # 预训练模型路径或 HuggingFace 模型 ID
    parser.add_argument(
        "--model_path",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Path or Hugging Face ID of the pretrained FLUX model."
    )
    # style_lambda 控制风格融合强度
    parser.add_argument(
        "--style_lambda",
        type=float,
        default=1.1,
        help="The lambda value for controlling style alignment strength."
    )

    # 英文提示词示例
    en_prompts = [
        # "Anchor in 3D realism style.",
        # "Clock in 3D realism style.",
        "Apple in 3D realism style.",
        "iPhone in 3D realism style.",
    ]

    # 中文提示词示例
    zh_prompts = [
        "3D写实风格的地球仪",
        "3D写实风格的iPhone",
    ]

    # 选择要使用的提示词
    prompts = en_prompts

    parsed_args = parser.parse_args()
    main(parsed_args)
