# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入必要的标准库和类型提示
import inspect
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

# 导入第三方库
import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast  # Transformers 库的文本处理模型

# 导入 Diffusers 库的核心组件
from diffusers.image_processor import VaeImageProcessor  # VAE 图像处理器
from diffusers.loaders import FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin  # 模型加载混入类
from diffusers.models.autoencoders import AutoencoderKL  # 自编码器模型
from diffusers.models.transformers import FluxTransformer2DModel  # Flux Transformer 模型
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  # 流匹配欧拉离散调度器
from diffusers.utils import (
    USE_PEFT_BACKEND,  # PEFT 后端支持标志
    is_torch_xla_available,  # XLA 可用性检查
    logging,
    replace_example_docstring,  # 替换示例文档字符串
    scale_lora_layers,  # LoRA 层缩放
    unscale_lora_layers,  # LoRA 层反缩放
)
from diffusers.utils.torch_utils import randn_tensor  # 随机数张量生成
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # 扩散管道基类
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput  # Flux 管道输出类

# 检查是否支持 XLA（PyTorch 的加速库，主要用于 TPU）
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # XLA 核心模块
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，展示如何使用 FluxPipeline
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxPipeline

        # 从预训练模型加载 Flux 管道
        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # 根据使用的变体，管道调用会略有不同
        >>> # 请参考管道文档了解更多详情
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        >>> image.save("flux.png")
        ```
"""


def calculate_shift(
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.16,
):
    """
    计算时间步偏移参数

    这个函数根据图像序列长度计算一个线性插值的偏移值，用于调整扩散模型中的时间步调度。
    偏移值会在基础序列长度和最大序列长度之间线性变化。

    Args:
        image_seq_len: 输入图像的序列长度
        base_seq_len (int): 基础序列长度，默认为 256
        max_seq_len (int): 最大序列长度，默认为 4096
        base_shift (float): 基础偏移值，默认为 0.5
        max_shift (float): 最大偏移值，默认为 1.16

    Returns:
        mu: 计算得到的偏移值，用于调整时间步调度
    """
    # 计算线性插值的斜率
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    # 计算线性插值的截距
    b = base_shift - m * base_seq_len
    # 根据当前图像序列长度计算偏移值
    mu = image_seq_len * m + b
    return mu


# 从 Stable Diffusion 管道复制的获取时间步函数
def retrieve_timesteps(
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
):
    """
    调用调度器的 set_timesteps 方法并从调度器中检索时间步

    处理自定义时间步。任何关键字参数都将提供给 scheduler.set_timesteps。

    Args:
        scheduler (SchedulerMixin): 要从中获取时间步的调度器
        num_inference_steps (int): 使用预训练模型生成样本时使用的扩散步数。如果使用，timesteps 必须为 None
        device (str or torch.device, optional): 时间步应该移动到的设备。如果为 None，时间步不会被移动
        timesteps (List[int], optional): 用于覆盖调度器时间步间距策略的自定义时间步。如果传递 timesteps，num_inference_steps 和 sigmas 必须为 None
        sigmas (List[float], optional): 用于覆盖调度器时间步间距策略的自定义 sigma 值。如果传递 sigmas，num_inference_steps 和 timesteps 必须为 None

    Returns:
        Tuple[torch.Tensor, int]: 元组，第一个元素是调度器的时间步调度，第二个元素是推理步数
    """
    # 检查不能同时传入 timesteps 和 sigmas
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")

    # 处理自定义时间步
    if timesteps is not None:
        # 检查调度器是否支持自定义时间步
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 设置自定义时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)

    # 处理自定义 sigma 值
    elif sigmas is not None:
        # 检查调度器是否支持自定义 sigma 值
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 设置自定义 sigma 值
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)

    # 使用默认设置
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


class FluxPipeline(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
):
    """
    Flux 文本到图像生成管道

    参考文档: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        transformer (FluxTransformer2DModel): 条件 Transformer (MMDiT) 架构，用于去噪编码的图像潜变量
        scheduler (FlowMatchEulerDiscreteScheduler): 与 transformer 结合使用去噪编码图像潜变量的调度器
        vae (AutoencoderKL): 变分自编码器模型，用于将图像编码到潜变量表示或从潜变量表示解码图像
        text_encoder (CLIPTextModel): CLIP 文本编码器，具体是 clip-vit-large-patch14 变体
        text_encoder_2 (T5EncoderModel): T5 文本编码器，具体是 google/t5-v1_1-xxl 变体
        tokenizer (CLIPTokenizer): CLIP 分词器类
        tokenizer_2 (T5TokenizerFast): T5 快速分词器类
    """

    # CPU 卸载序列：定义模型组件从 CPU 卸载到 GPU 的顺序
    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    # 可选组件列表
    _optional_components = []
    # 回调函数张量输入列表
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
            self,
            scheduler: FlowMatchEulerDiscreteScheduler,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            text_encoder_2: T5EncoderModel,
            tokenizer_2: T5TokenizerFast,
            transformer: FluxTransformer2DModel,
    ):
        """
        初始化 FluxPipeline 实例

        Args:
            scheduler: 流匹配调度器
            vae: 变分自编码器
            text_encoder: CLIP 文本编码器
            tokenizer: CLIP 分词器
            text_encoder_2: T5 文本编码器
            tokenizer_2: T5 分词器
            transformer: Flux Transformer 模型
        """
        super().__init__()

        # 注册所有模型组件
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
        )

        # 计算 VAE 缩放因子
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )

        # Flux 潜变量被转换为 2x2 补丁并打包，这意味着潜在宽度和高度必须能被补丁大小整除
        # 因此 VAE 缩放因子乘以补丁大小以考虑这一点
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

        # 获取分词器的最大长度
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )

        # 设置默认采样大小
        self.default_sample_size = 128

    def _get_t5_prompt_embeds(
            self,
            prompt: Union[str, List[str]] = None,
            num_images_per_prompt: int = 1,
            max_sequence_length: int = 512,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        """
        获取 T5 文本编码器的提示嵌入

        Args:
            prompt: 输入提示词（字符串或字符串列表）
            num_images_per_prompt: 每个提示生成的图像数量
            max_sequence_length: 最大序列长度
            device: 计算设备
            dtype: 数据类型

        Returns:
            torch.Tensor: T5 提示嵌入张量
        """
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        # 将单个提示词转换为列表形式
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        # 处理文本反转（Textual Inversion）
        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        # 使用 T5 分词器编码文本
        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        # 检查未截断的输入以确定是否有内容被截断
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1: -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        # 通过 T5 编码器获取文本嵌入
        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        # 确保数据类型和设备正确
        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # 为每个生成重复文本嵌入，使用 MPS 友好的方法
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
            self,
            prompt: Union[str, List[str]],
            num_images_per_prompt: int = 1,
            device: Optional[torch.device] = None,
    ):
        """
        获取 CLIP 文本编码器的提示嵌入

        Args:
            prompt: 输入提示词（字符串或字符串列表）
            num_images_per_prompt: 每个提示生成的图像数量
            device: 计算设备

        Returns:
            torch.Tensor: CLIP 提示嵌入张量（池化输出）
        """
        device = device or self._execution_device

        # 将单个提示词转换为列表形式
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        # 处理文本反转（Textual Inversion）
        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        # 使用 CLIP 分词器编码文本
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids

        # 检查未截断的输入以确定是否有内容被截断
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1: -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )

        # 通过 CLIP 编码器获取文本嵌入
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # 使用 CLIPTextModel 的池化输出
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # 为每个生成重复文本嵌入，使用 MPS 友好的方法
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
            self,
            prompt: Union[str, List[str]],
            prompt_2: Union[str, List[str]],
            device: Optional[torch.device] = None,
            num_images_per_prompt: int = 1,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            max_sequence_length: int = 512,
            lora_scale: Optional[float] = None,
    ):
        """
        编码提示词，同时使用 CLIP 和 T5 文本编码器

        Args:
            prompt (str or List[str], optional): 要编码的主要提示词
            prompt_2 (str or List[str], optional): 发送到 tokenizer_2 和 text_encoder_2 的提示词，如果未定义则使用 prompt
            device (torch.device): torch 设备
            num_images_per_prompt (int): 每个提示应生成的图像数量
            prompt_embeds (torch.FloatTensor, optional): 预生成的文本嵌入，可以用来轻松调整文本输入，如提示词权重
            pooled_prompt_embeds (torch.FloatTensor, optional): 预生成的池化文本嵌入
            max_sequence_length (int): 最大序列长度
            lora_scale (float, optional): 如果加载了 LoRA 层，将应用于文本编码器所有 LoRA 层的 LoRA 缩放因子

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 包含提示嵌入、池化提示嵌入和文本 ID 的元组
        """
        device = device or self._execution_device

        # 设置 LoRA 缩放因子，以便文本编码器的猴子补丁 LoRA 函数可以正确访问它
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # 动态调整 LoRA 缩放
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        # 将单个提示词转换为列表形式
        prompt = [prompt] if isinstance(prompt, str) else prompt

        # 如果没有提供预生成的提示嵌入，则生成它们
        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt  # 如果未提供 prompt_2，使用 prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # 只使用 CLIPTextModel 的池化提示输出
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )

            # 获取 T5 提示嵌入
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        # 恢复 CLIP 文本编码器的原始 LoRA 缩放
        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # 通过缩放恢复 LoRA 层的原始缩放
                unscale_lora_layers(self.text_encoder, lora_scale)

        # 恢复 T5 文本编码器的原始 LoRA 缩放
        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # 通过缩放恢复 LoRA 层的原始缩放
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        # 确定数据类型并创建文本 ID
        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def check_inputs(
            self,
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            callback_on_step_end_tensor_inputs=None,
            max_sequence_length=None,
    ):
        """
        检查输入参数的有效性

        Args:
            prompt: 主要提示词
            prompt_2: 次要提示词
            height: 图像高度
            width: 图像宽度
            prompt_embeds: 预生成的提示嵌入
            pooled_prompt_embeds: 预生成的池化提示嵌入
            callback_on_step_end_tensor_inputs: 步骤结束回调的张量输入
            max_sequence_length: 最大序列长度

        Raises:
            ValueError: 当输入参数不符合要求时抛出异常
            Warning: 当高度或宽度需要调整时发出警告
        """
        # 检查高度和宽度是否能被 vae_scale_factor * 2 整除
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        # 检查回调张量输入是否在允许的列表中
        if callback_on_step_end_tensor_inputs is not None and not all(
                k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # 检查提示词和提示嵌入的冲突
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        # 检查如果提供了提示嵌入，是否也提供了池化提示嵌入
        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        # 检查最大序列长度是否超过限制
        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        """
        准备潜变量图像 ID

        为每个潜变量位置创建唯一的 ID 张量，用于在 Transformer 模型中表示位置信息。

        Args:
            batch_size: 批次大小（虽然在此函数中未直接使用）
            height: 图像高度
            width: 图像宽度
            device: 计算设备
            dtype: 数据类型

        Returns:
            torch.Tensor: 形状为 (height*width, 3) 的图像 ID 张量
        """
        # 创建一个 3 通道的零张量，每个位置对应一个唯一的 ID
        latent_image_ids = torch.zeros(height, width, 3)

        # 第 1 通道存储行（高度）位置信息
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]

        # 第 2 通道存储列（宽度）位置信息
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        # 获取原始形状
        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        # 重塑为 (高度*宽度, 通道数) 的二维张量
        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        # 移动到指定设备并转换数据类型
        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        """
        打包潜变量

        将潜变量张量重新排列为 2x2 补丁并打包，这是 Flux 模型的特殊要求。
        将 (batch_size, channels, height, width) 转换为 (batch_size, num_patches, channels*4)

        Args:
            latents: 输入潜变量张量
            batch_size: 批次大小
            num_channels_latents: 潜变量通道数
            height: 原始高度
            width: 原始宽度

        Returns:
            torch.Tensor: 打包后的潜变量张量
        """
        # 将潜变量重塑为包含 2x2 补丁的六维张量
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)

        # 重新排列维度：(batch, patch_h, patch_w, channels, 2, 2) -> (batch, patch_h, patch_w, channels, 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)

        # 重塑为 (batch_size, num_patches, channels*4)，其中每个 2x2 补丁的 4 个像素被展平
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        """
        解包潜变量

        将打包的潜变量重新排列回原始形状，与 _pack_latents 相反的操作。
        将 (batch_size, num_patches, channels*4) 转换回 (batch_size, channels, height, width)

        Args:
            latents: 打包的潜变量张量
            height: 目标高度
            width: 目标宽度
            vae_scale_factor: VAE 缩放因子

        Returns:
            torch.Tensor: 解包后的潜变量张量
        """
        batch_size, num_patches, channels = latents.shape

        # VAE 对图像应用 8x 压缩，但我们必须考虑打包，这要求潜在高度和宽度能被 2 整除
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        # 重塑为六维张量以恢复 2x2 补丁结构
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)

        # 重新排列维度以恢复原始通道优先顺序
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        # 重塑回标准的 (batch_size, channels, height, width) 格式
        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    def enable_vae_slicing(self):
        """
        启用 VAE 分片解码

        当启用此选项时，VAE 会将输入张量分割成片来分步计算解码。
        这对于节省一些内存和允许更大的批次大小很有用。
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        """
        禁用 VAE 分片解码

        如果之前启用了 enable_vae_slicing，此方法将返回到一步计算解码。
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        """
        启用 VAE 平铺解码

        当启用此选项时，VAE 会将输入张量分割成平铺块来分步计算解码和编码。
        这对于节省大量内存和允许处理更大的图像很有用。
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        """
        禁用 VAE 平铺解码

        如果之前启用了 enable_vae_tiling，此方法将返回到一步计算解码。
        """
        self.vae.disable_tiling()

    def prepare_latents(
            self,
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents=None,
    ):
        """
        准备初始潜变量

        生成或准备用于扩散过程的初始噪声潜变量。

        Args:
            batch_size: 批次大小
            num_channels_latents: 潜变量通道数
            height: 目标图像高度
            width: 目标图像宽度
            dtype: 数据类型
            device: 计算设备
            generator: 随机数生成器
            latents: 可选的预生成潜变量

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 包含准备好的潜变量和潜变量图像 ID 的元组

        Raises:
            ValueError: 当生成器列表长度与批次大小不匹配时
        """
        # VAE 对图像应用 8x 压缩，但我们必须考虑打包，这要求潜在高度和宽度能被 2 整除
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        # 定义潜变量的形状
        shape = (batch_size, num_channels_latents, height, width)

        # 如果提供了预生成的潜变量，直接使用
        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        # 检查生成器列表长度是否匹配批次大小
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 生成随机噪声潜变量
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # 打包潜变量以适配 Flux 模型的要求
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        # 准备潜变量图像 ID
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids

    @property
    def guidance_scale(self):
        """获取指导缩放因子"""
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        """获取联合注意力参数"""
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        """获取时间步数量"""
        return self._num_timesteps

    @property
    def interrupt(self):
        """获取中断标志"""
        return self._interrupt

    @torch.no_grad()  # 禁用梯度计算以提高推理效率
    @replace_example_docstring(EXAMPLE_DOC_STRING)  # 替换示例文档字符串
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,  # 指导图像生成的提示词
            prompt_2: Optional[Union[str, List[str]]] = None,  # 发送到第二个文本编码器的提示词
            height: Optional[int] = None,  # 生成图像的高度（像素）
            width: Optional[int] = None,  # 生成图像的宽度（像素）
            num_inference_steps: int = 28,  # 去噪步数
            timesteps: List[int] = None,  # 自定义时间步
            guidance_scale: float = 3.5,  # 指导缩放因子
            num_images_per_prompt: Optional[int] = 1,  # 每个提示生成的图像数量
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 随机数生成器
            latents: Optional[torch.FloatTensor] = None,  # 预生成的噪声潜变量
            prompt_embeds: Optional[torch.FloatTensor] = None,  # 预生成的文本嵌入
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,  # 预生成的池化文本嵌入
            output_type: Optional[str] = "pil",  # 输出格式："pil" 或 "latent"
            return_dict: bool = True,  # 是否返回字典格式的结果
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,  # 联合注意力参数
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,  # 步骤结束回调函数
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],  # 回调函数的张量输入
            max_sequence_length: int = 512,  # 最大序列长度
            use_same_latent: bool = False,  # 是否使用相同的初始噪声
    ):
        """
        调用管道进行生成时执行的函数
        Examples:

        Returns:
            FluxPipelineOutput 或 tuple: 根据 return_dict 参数返回相应的结果格式
        """
            
        # 设置默认的高度和宽度
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. 检查输入参数，如果不正确则抛出错误
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        # 设置内部状态变量
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. 定义调用参数
        # 确定批次大小
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 获取执行设备
        device = self._execution_device

        # 获取 LoRA 缩放因子
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        # 3. 编码提示词
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. 准备潜变量
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        # 如果指定使用相同的初始噪声（用于一致性比较）
        if use_same_latent:
            print(f"Using same initial noise! pre latents shape:{latents.shape}")
            bs_ = latents.shape[0]
            latents = latents[0].repeat(bs_, 1, 1, )
            print(f"process latents shape:{latents.shape}")

        # ****************************************************************************************
        # 创建参考图像 ID（特殊功能，用于某种条件生成）
        reference_image_ids = deepcopy(latent_image_ids)
        reference_image_ids[:, 2] += width // 16  # 在宽度维度上偏移
        # ****************************************************************************************

        # 5. 准备时间步
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 处理指导嵌入
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 6. 去噪循环
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # 检查中断标志
                if self.interrupt:
                    continue

                # ****************************************************************************************
                # 为所有注意力处理器设置当前时间步（特殊功能）
                for id, name in enumerate(self.transformer.attn_processors.keys()):
                    self.transformer.attn_processors[name].set_timesteps(i, t)
                # ****************************************************************************************

                # 广播到批次维度，与 ONNX/Core ML 兼容
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # 通过 Transformer 模型预测噪声
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,  # 归一化时间步
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                    # ****************************************************************************************
                    concat_img_ids=reference_image_ids,  # 传入参考图像 ID
                    # ****************************************************************************************
                )[0]

                # 计算上一个噪声样本 x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # 处理数据类型问题（MPS 平台兼容性）
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # 某些平台（如 apple mps）由于 pytorch bug 而行为异常
                        latents = latents.to(latents_dtype)

                # 处理步骤结束回调
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # 更新进度条
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                # XLA 步骤标记（用于 TPU）
                if XLA_AVAILABLE:
                    xm.mark_step()

        # 处理输出
        if output_type == "latent":
            image = latents
        else:
            # 解包潜变量并解码图像
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # 卸载所有模型以释放内存
        self.maybe_free_model_hooks()

        # 根据返回格式参数返回结果
        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
