# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
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

"""
Flux Transformer模型实现
=======================

本文件实现了Flux扩散模型的核心Transformer架构，包含：
1. FluxSingleTransformerBlock - 单流Transformer块（仅处理图像）
2. FluxTransformerBlock - 双流Transformer块（同时处理图像和文本）
3. FluxTransformer2DModel - 主要的2D Transformer模型

主要特性：
- MMDiT（Multimodal Diffusion Transformer）架构
- 支持双流和单流Transformer处理
- 集成ControlNet支持
- 支持梯度检查点以节省内存
- 支持LoRA适配器
- 旋转位置编码（RoPE）
- 自适应层归一化
"""

from diffusers import UNet2DConditionModel
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    FluxAttnProcessor2_0,
    FluxAttnProcessor2_0_NPU,
    FusedFluxAttnProcessor2_0,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, \
    FluxPosEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from .attention_processor import StyleAlignedArgs

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):
    r"""
    遵循MMDiT架构的Transformer块，引入自Stable Diffusion 3。
    这是单流Transformer块，只处理图像特征。

    参考文献: https://arxiv.org/abs/2403.03206

    参数:
        dim (`int`): 输入和输出的通道数。
        num_attention_heads (`int`): 多头注意力使用的头数。
        attention_head_dim (`int`): 每个头的通道数。
        mlp_ratio (`float`): MLP隐藏层维度与输入维度的比例，默认为4.0。
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0):
        super().__init__()
        # 计算MLP隐藏层维度
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        # 自适应层归一化，只处理单一输入
        self.norm = AdaLayerNormZeroSingle(dim)

        # MLP相关层
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)  # 输入投影层
        self.act_mlp = nn.GELU(approximate="tanh")  # 激活函数
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)  # 输出投影层

        # 选择注意力处理器：支持NPU或标准处理器
        if is_torch_npu_available():
            processor = FluxAttnProcessor2_0_NPU()  # NPU专用处理器
        else:
            processor = FluxAttnProcessor2_0()  # 标准处理器

        # 自注意力机制
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,  # 不使用交叉注意力
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",  # 使用RMS归一化
            eps=1e-6,
            pre_only=True,  # 仅前向处理
        )

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            image_rotary_emb=None,
            joint_attention_kwargs=None,
            image_rotary_emb_additional=None,
            txt_length=None,
    ):
        """
        FluxSingleTransformerBlock的前向传播方法

        参数:
            hidden_states: 输入的隐藏状态张量 [B, L, D]
            temb: 时间嵌入张量 [B, D]，用于自适应归一化
            image_rotary_emb: 图像旋转位置编码
            joint_attention_kwargs: 注意力参数
            image_rotary_emb_additional: 额外的图像旋转位置编码
            txt_length: 文本长度（此处未使用，保持接口一致性）
        """
        # 保存残差连接
        residual = hidden_states

        # 自适应归一化：返回归一化后的状态和门控值
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

        # MLP分支：通过线性层和激活函数
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        # 确保注意力参数不为空
        joint_attention_kwargs = joint_attention_kwargs or {}

        # 自注意力计算
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            image_rotary_emb_additional=image_rotary_emb_additional,
            txt_length=txt_length,
            **joint_attention_kwargs,
        )

        # 连接注意力输出和MLP输出
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)

        # 应用门控：扩展门控维度并应用
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)

        # 添加残差连接
        hidden_states = residual + hidden_states

        # 对float16类型进行裁剪以防止数值溢出
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


@maybe_allow_in_graph
class FluxTransformerBlock(nn.Module):
    r"""
    遵循MMDiT架构的Transformer块，引入自Stable Diffusion 3。
    这是双流Transformer块，同时处理图像和文本特征。

    参考文献: https://arxiv.org/abs/2403.03206

    参数:
        dim (`int`): 输入和输出的通道数。
        num_attention_heads (`int`): 多头注意力使用的头数。
        attention_head_dim (`int`): 每个头的通道数。
        qk_norm (`str`): 查询和键的归一化方法，默认为"rms_norm"。
        eps (`float`): 数值稳定性参数，默认为1e-6。
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6):
        super().__init__()

        # 图像流的第一层归一化
        self.norm1 = AdaLayerNormZero(dim)

        # 文本流的第一层归一化（上下文）
        self.norm1_context = AdaLayerNormZero(dim)

        # 检查PyTorch是否支持高效的缩放点积注意力
        if hasattr(F, "scaled_dot_product_attention"):
            processor = FluxAttnProcessor2_0()  # 使用高效注意力处理器
        else:
            raise ValueError(
                "当前PyTorch版本不支持`scaled_dot_product_attention`函数。"
            )

        # 双流注意力机制：同时处理图像和文本
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,         # 标准交叉注意力维度
            added_kv_proj_dim=dim,            # 添加的键值投影维度（用于双流）
            dim_head=attention_head_dim,      # 每个注意力头的维度
            heads=num_attention_heads,        # 注意力头数量
            out_dim=dim,
            context_pre_only=False,           # 不仅预处理上下文
            bias=True,                        # 使用偏置
            processor=processor,              # 注意力处理器
            qk_norm=qk_norm,                  # 查询键归一化方法
            eps=eps,                          # 数值稳定性参数
        )

        # 图像流的第二层归一化和前馈网络
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # 文本流的第二层归一化和前馈网络
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # 分块处理参数（用于内存优化）
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            image_rotary_emb=None,
            joint_attention_kwargs=None,
            image_rotary_emb_additional=None,
            txt_length=None,
    ):
        """
        FluxTransformerBlock的前向传播方法（双流处理）

        参数:
            hidden_states: 图像隐藏状态张量 [B, L_img, D]
            encoder_hidden_states: 文本编码器隐藏状态张量 [B, L_txt, D]
            temb: 时间嵌入张量 [B, D]，用于自适应归一化
            image_rotary_emb: 图像旋转位置编码
            joint_attention_kwargs: 注意力参数
            image_rotary_emb_additional: 额外的图像旋转位置编码
            txt_length: 文本长度

        返回:
            tuple: (encoder_hidden_states, hidden_states) 处理后的文本和图像状态
        """
        # ===== 第一阶段：图像流处理 =====
        # 图像流的自适应归一化，返回多个门控参数
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        # ===== 第二阶段：文本流处理 =====
        # 文本流的自适应归一化
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # 确保注意力参数不为空
        joint_attention_kwargs = joint_attention_kwargs or {}

        # ===== 第三阶段：双流注意力计算 =====
        # 同时处理图像和文本的注意力
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,           # 图像查询
            encoder_hidden_states=norm_encoder_hidden_states,  # 文本键值
            image_rotary_emb=image_rotary_emb,
            image_rotary_emb_additional=image_rotary_emb_additional,
            txt_length=txt_length,
            **joint_attention_kwargs,
        )

        # ===== 第四阶段：处理图像流的注意力输出 =====
        # 应用注意力门控
        attn_output = gate_msa.unsqueeze(1) * attn_output
        # 残差连接
        hidden_states = hidden_states + attn_output

        # 图像流的前馈网络处理
        norm_hidden_states = self.norm2(hidden_states)
        # 应用自适应缩放和平移
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        # 前馈网络计算
        ff_output = self.ff(norm_hidden_states)
        # 应用MLP门控
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        # 添加残差连接
        hidden_states = hidden_states + ff_output

        # ===== 第五阶段：处理文本流的注意力输出 =====
        # 应用注意力门控
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        # 残差连接
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        # 文本流的前馈网络处理
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        # 应用自适应缩放和平移
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        # 前馈网络计算
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        # 应用MLP门控并残差连接
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        # 对float16类型进行裁剪以防止数值溢出
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class FluxTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    Flux引入的Transformer模型。

    这是Flux扩散模型的核心架构，结合了双流Transformer（处理图像和文本）和单流Transformer（仅处理图像）。

    参考文献: https://blackforestlabs.ai/announcing-black-forest-labs/

    参数:
        patch_size (`int`): 将输入数据转换为小块的尺寸，默认为1。
        in_channels (`int`, *可选*, 默认为16): 输入的通道数。
        out_channels (`int`, *可选*): 输出的通道数，默认与in_channels相同。
        num_layers (`int`, *可选*, 默认为19): 使用的MMDiT块层数。
        num_single_layers (`int`, *可选*, 默认为38): 使用的单流DiT块层数。
        attention_head_dim (`int`, *可选*, 默认为128): 每个注意力头的通道数。
        num_attention_heads (`int`, *可选*, 默认为24): 多头注意力使用的头数。
        joint_attention_dim (`int`, *可选*): 编码器隐藏状态的维度，默认为4096。
        pooled_projection_dim (`int`): 池化投影的维度，默认为768。
        guidance_embeds (`bool`, 默认为False): 是否使用引导嵌入。
        axes_dims_rope (`tuple`, 默认为(16, 56, 56)): 旋转位置编码的轴维度。
    """

    # 支持梯度检查点以节省内存
    _supports_gradient_checkpointing = True

    # 定义哪些模块不应该被分割（用于模型并行）
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    @register_to_config
    def __init__(
            self,
            patch_size: int = 1,
            in_channels: int = 64,
            out_channels: Optional[int] = None,
            num_layers: int = 19,
            num_single_layers: int = 38,
            attention_head_dim: int = 128,
            num_attention_heads: int = 24,
            joint_attention_dim: int = 4096,
            pooled_projection_dim: int = 768,
            guidance_embeds: bool = False,
            axes_dims_rope: Tuple[int] = (16, 56, 56),
    ):
        """
        FluxTransformer2DModel的初始化方法

        参数说明详见类文档字符串。
        """
        super().__init__()

        # 设置输出通道数（默认与输入相同）
        self.out_channels = out_channels or in_channels

        # 计算内部维度：注意力头数 × 每个头的维度
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        # 旋转位置编码器
        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        # 根据是否使用引导嵌入选择时间文本嵌入类
        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        # 时间和文本的嵌入层
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )

        # 上下文嵌入器：将文本编码投影到模型维度
        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)

        # 输入嵌入器：将图像输入投影到模型维度
        self.x_embedder = nn.Linear(self.config.in_channels, self.inner_dim)

        # ===== 双流Transformer块（处理图像和文本） =====
        # 创建多层双流Transformer块
        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_layers)
            ]
        )

        # ===== 单流Transformer块（仅处理图像） =====
        # 创建多层单流Transformer块
        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        # 输出归一化层
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)

        # 输出投影层：将特征投影到输出空间
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        # 梯度检查点标志（默认关闭以节省内存）
        self.gradient_checkpointing = False

    @property
    # 从diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors复制
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        """
        返回模型中所有注意力处理器的字典

        返回:
            Dict[str, AttentionProcessor]: 包含模型中所有注意力处理器的字典，
                以权重名称为索引。
        """
        # 递归设置处理器
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            """递归添加注意力处理器"""
            # 如果模块有get_processor方法，添加到字典中
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            # 递归处理所有子模块
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        # 处理所有顶级子模块
        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # 从diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor复制
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        """
        设置用于计算注意力的处理器

        参数:
            processor (`AttentionProcessor` 或 `AttentionProcessor` 的字典):
                要设置为所有注意力层处理器的实例化类。

                如果 `processor` 是字典，键需要定义到对应交叉注意力处理器的路径。
                在设置可训练的注意力处理器时强烈推荐使用字典方式。

        异常:
            ValueError: 当传递的处理器数量与注意力层数量不匹配时抛出
        """
        # 获取注意力处理器的数量
        count = len(self.attn_processors.keys())

        # 验证处理器数量匹配
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"传递了一个处理器字典，但处理器数量{len(processor)}与注意力层数量{count}不匹配。"
                f"请确保传递{count}个处理器类。"
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            """递归设置注意力处理器"""
            # 如果模块有set_processor方法，设置处理器
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    # 单个处理器应用于所有层
                    module.set_processor(processor)
                else:
                    # 从字典中获取对应的处理器
                    module.set_processor(processor.pop(f"{name}.processor"))

            # 递归处理所有子模块
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        # 处理所有顶级子模块
        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # 从diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections复制，使用FusedFluxAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        启用融合的QKV投影。对于自注意力模块，所有投影矩阵（查询、键、值）都被融合。
        对于交叉注意力模块，键和值投影矩阵被融合。

        <Tip warning={true}>

        这个API是 实验性的。

        </Tip>
        """
        # 保存原始注意力处理器
        self.original_attn_processors = None

        # 检查是否支持融合
        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("对于有添加KV投影的模型，不支持`fuse_qkv_projections()`。")

        # 备份当前处理器
        self.original_attn_processors = self.attn_processors

        # 对所有注意力模块启用投影融合
        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        # 设置融合的注意力处理器
        self.set_attn_processor(FusedFluxAttnProcessor2_0())

    # 从diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections复制
    def unfuse_qkv_projections(self):
        """
        如果启用了融合QKV投影，则禁用它。

        <Tip warning={true}>

        这个API是 实验性的。

        </Tip>

        """
        # 如果有备份的原始处理器，恢复它们
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _set_gradient_checkpointing(self, module, value=False):
        """
        递归设置模块的梯度检查点

        参数:
            module: 要设置的模块
            value: 是否启用梯度检查点，默认为False
        """
        # 如果模块有gradient_checkpointing属性，设置其值
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def set_style_aligned_args(self, style_aligned_args: StyleAlignedArgs):
        """
        设置风格对齐参数

        参数:
            style_aligned_args: 风格对齐参数对象
        """
        self.style_aligned_args = style_aligned_args

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            img_ids: torch.Tensor = None,
            txt_ids: torch.Tensor = None,
            guidance: torch.Tensor = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_block_samples=None,
            controlnet_single_block_samples=None,
            return_dict: bool = True,
            controlnet_blocks_repeat: bool = False,
            concat_img_ids: torch.Tensor = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        FluxTransformer2DModel的前向传播方法

        参数:
            hidden_states (`torch.FloatTensor`，形状为`(batch_size, channel, height, width)`):
                输入的隐藏状态张量，通常是噪声图像的潜在表示
            encoder_hidden_states (`torch.FloatTensor`，形状为`(batch_size, sequence_len, embed_dims)`):
                条件嵌入，从输入条件（如提示词）计算得到的嵌入向量
            pooled_projections (`torch.FloatTensor`，形状为`(batch_size, projection_dim)`):
                从输入条件嵌入投影得到的嵌入向量
            timestep (`torch.LongTensor`): 用于表示去噪步骤的时间步
            img_ids (`torch.Tensor`): 图像的位置ID，用于位置编码
            txt_ids (`torch.Tensor`): 文本的位置ID，用于位置编码
            guidance (`torch.Tensor`): 引导信号，用于条件生成
            joint_attention_kwargs (`dict`, *可选*): 传递给注意力处理器的参数字典
            controlnet_block_samples (`list` of `torch.Tensor`): ControlNet块样本，如果指定则添加到transformer块的残差中
            controlnet_single_block_samples (`list` of `torch.Tensor`): ControlNet单块样本
            return_dict (`bool`, *可选*, 默认为`True`): 是否返回Transformer2DModelOutput对象而不是元组
            controlnet_blocks_repeat (`bool`): 是否重复使用ControlNet块样本
            concat_img_ids (`torch.Tensor`): 连接的图像ID

        返回:
            如果`return_dict`为True，返回[`~models.transformer_2d.Transformer2DModelOutput`]，
            否则返回元组，第一个元素是样本张量。
        """
        # ===== 第一阶段：参数预处理和LoRA缩放 =====
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            # 从注意力参数中提取LoRA缩放因子
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        # 如果使用PEFT后端，应用LoRA缩放
        if USE_PEFT_BACKEND:
            # 通过为每个PEFT层设置`lora_scale`来加权LoRA层
            scale_lora_layers(self, lora_scale)
        else:
            # 如果不使用PEFT后端但传递了缩放参数，发出警告
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "在不使用PEFT后端时通过`joint_attention_kwargs`传递`scale`是无效的。"
                )

        # ===== 第二阶段：输入嵌入 =====
        # 将输入隐藏状态投影到模型维度
        hidden_states = self.x_embedder(hidden_states)

        # 时间步处理：转换为与隐藏状态相同的数据类型并缩放
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            # 引导信号也进行相同的处理
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        # 时间嵌入生成：结合时间步、池化投影和引导信号
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )

        # 编码器隐藏状态投影：将文本编码投影到模型维度
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # ===== 第三阶段：位置编码处理 =====
        # 检查并处理过时的文本ID格式
        if txt_ids.ndim == 3:
            logger.warning(
                "传递3维`txt_ids`张量已弃用。"
                "请移除批次维度并将其作为2维张量传递"
            )
            txt_ids = txt_ids[0]

        # 检查并处理过时的图像ID格式
        if img_ids.ndim == 3:
            logger.warning(
                "传递3维`img_ids`张量已弃用。"
                "请移除批次维度并将其作为2维张量传递"
            )
            img_ids = img_ids[0]

        # 连接文本和图像ID以生成位置编码
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)
        # 为额外的连接图像生成位置编码
        image_rotary_emb_additional = self.pos_embed(concat_img_ids)

        # ===== 第四阶段：双流Transformer处理 =====
        # 记录文本长度用于后续处理
        txt_length = encoder_hidden_states.shape[1]

        # 遍历所有双流Transformer块
        for index_block, block in enumerate(self.transformer_blocks):
            # 如果启用梯度检查点且梯度计算开启
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    """创建自定义前向传播函数以用于梯度检查点"""
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward

                # 根据PyTorch版本设置检查点参数
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

                # 使用梯度检查点进行前向传播（节省内存）
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                # 标准前向传播
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    image_rotary_emb_additional=image_rotary_emb_additional,
                    txt_length=txt_length,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # ===== ControlNet残差处理 =====
            # 如果提供了ControlNet块样本，添加到隐藏状态中
            if controlnet_block_samples is not None:
                # 计算控制采样间隔
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))

                # 对于Xlabs ControlNet，根据重复模式添加残差
                if controlnet_blocks_repeat:
                    # 重复模式：循环使用ControlNet样本
                    hidden_states = (
                            hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    # 非重复模式：按间隔分配ControlNet样本
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        # ===== 第五阶段：单流Transformer处理 =====
        # 重新记录文本长度（经过双流处理后可能改变）
        txt_length = encoder_hidden_states.shape[1]

        # 连接文本和图像隐藏状态用于单流处理
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # 遍历所有单流Transformer块
        for index_block, block in enumerate(self.single_transformer_blocks):
            # 如果启用梯度检查点且梯度计算开启
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    """创建自定义前向传播函数以用于梯度检查点"""
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward

                # 根据PyTorch版本设置检查点参数
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

                # 使用梯度检查点进行前向传播（节省内存）
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                # 标准前向传播
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    image_rotary_emb_additional=image_rotary_emb_additional,
                    txt_length=txt_length,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # ===== 单流ControlNet残差处理 =====
            # 如果提供了单流ControlNet块样本，添加到对应的图像部分
            if controlnet_single_block_samples is not None:
                # 计算控制采样间隔
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))

                # 只对图像部分添加ControlNet残差（跳过文本部分）
                hidden_states[:, encoder_hidden_states.shape[1]:, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1]:, ...]
                        + controlnet_single_block_samples[index_block // interval_control]
                )

        # ===== 第六阶段：输出处理 =====
        # 分离出图像部分的隐藏状态（移除文本部分）
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]

        # 最终归一化
        hidden_states = self.norm_out(hidden_states, temb)

        # 输出投影：将特征投影到输出空间
        output = self.proj_out(hidden_states)

        # ===== LoRA后处理 =====
        if USE_PEFT_BACKEND:
            # 从每个PEFT层移除`lora_scale`
            unscale_lora_layers(self, lora_scale)

        # ===== 返回结果 =====
        if not return_dict:
            # 返回元组格式
            return (output,)

        # 返回Transformer2DModelOutput对象
        return Transformer2DModelOutput(sample=output)
