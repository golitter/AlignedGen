import torch
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention_processor import Attention
from dataclasses import dataclass
from diffusers.models.embeddings import apply_rotary_emb


@dataclass(frozen=True)
class StyleAlignedArgs:
    """风格对齐配置参数"""
    share_attention: bool = True
    block: tuple[int, int] = (19, 57)  # 应用的Block范围
    timesteps: tuple[int, int] = (0, 30)  # 应用的时间步范围
    style_lambda_mode: str = "decrease"  # 风格混合模式 ["decrease", "fix"]
    style_lambda: float = 1.  # 风格混合强度
    constrain_first: bool = True  # 是否约束第一个样本


T = torch.Tensor


def expand_first(feat: T, scale=1., ) -> T:
    """将第一个样本的特征扩展到整个batch"""
    bs = feat.shape[0]
    feat_style = feat[0].unsqueeze(0).repeat(bs, 1, 1, 1)
    return feat_style


def concat_first(feat: T, dim=2, scale=1.) -> T:
    """将第一个样本的特征拼接到原始特征上"""
    bs = feat.shape[0]
    feat_style = feat[0].unsqueeze(0).repeat(bs, 1, 1, 1)
    return torch.cat((feat, feat_style), dim=dim)


def concat_first_block(feat_all: T, feat_block: T, dim=2, scale=1.) -> T:
    """将第一个block的特征拼接到所有特征上"""
    bs = feat_all.shape[0]
    if scale == 1.:
        feat_style = feat_block[0].unsqueeze(0).repeat(bs, 1, 1, 1)
    else:
        feat_style = (scale * feat_block[0]).unsqueeze(0).repeat(bs - 1, 1, 1, 1)
        feat_style = torch.cat((feat_block[0].unsqueeze(0), feat_style), dim=0)
    return torch.cat((feat_all, feat_style), dim=dim)


def calc_mean_std(feat, eps: float = 1e-5) -> tuple[T, T]:
    """计算特征的均值和标准差"""
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std


def adain(feat: T) -> T:
    """自适应实例归一化，将第一个样本的风格应用到所有样本"""
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat


class ShareAttnFluxAttnProcessor2_0:
    """Flux模型的风格对齐注意力处理器，用于处理SD3类自注意力投影"""

    def __init__(self, cnt: int, style_aligned_args: StyleAlignedArgs):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.cnt = cnt  # 注意力层数计数器
        self.t = 1  # 当前时间步
        self.args = style_aligned_args  # 风格对齐参数
        self.attn_weights = []  # 存储注意力权重

    def set_timesteps(self, t, timesteps):
        """设置当前时间步并计算风格混合强度"""
        self.t = t
        if self.args.style_lambda_mode == "decrease":
            # 递减模式：随时间步减弱风格混合
            self.scale = (timesteps / 1000) * (timesteps / 1000)
        elif self.args.style_lambda_mode == "fix":
            # 固定模式：使用固定强度
            self.scale = self.args.style_lambda

    def set_args(self, style_aligned_args):
        """更新风格对齐参数"""
        self.args = style_aligned_args

    def ori_call(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        """原始的注意力计算方法（不进行风格对齐）"""
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # 样本投影
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # 重塑为多头注意力格式
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # 查询和键的归一化
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # FluxSingleTransformerBlock中的注意力不使用encoder_hidden_states
        if encoder_hidden_states is not None:
            # 编码器状态投影
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            # 编码器投影的归一化
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # 拼接查询、键、值
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        # 应用旋转位置编码
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # 计算缩放点积注意力
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # 如果有编码器状态，分离处理
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1]:],
            )

            # 线性投影
            hidden_states = attn.to_out[0](hidden_states)
            # Dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            image_rotary_emb_additional: Optional[torch.Tensor] = None,
            txt_length: int = None,
    ) -> torch.FloatTensor:
        """主要的注意力处理器调用方法，实现风格对齐功能"""
        # 如果不启用共享注意力，使用原始方法
        if not self.args.share_attention:
            return self.ori_call(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                image_rotary_emb,
            )
        # 检查是否在指定的Block范围和时间步范围内
        if not ((self.cnt >= self.args.block[0] and self.cnt < self.args.block[1])
                and (self.t >= self.args.timesteps[0] and self.t < self.args.timesteps[1])):
            return self.ori_call(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                image_rotary_emb,
            )

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # 样本投影
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # 重塑为多头注意力格式
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # 查询和键的归一化
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

              # =============================================================================
        # 将hidden_states分割为编码器状态和图像状态
        if encoder_hidden_states is None:
            encoder_hidden_states_q = query[:, :, :txt_length, :]
            encoder_hidden_states_k = key[:, :, :txt_length, :]
            encoder_hidden_states_v = value[:, :, :txt_length, :]
            query = query[:, :, txt_length:, :]
            key = key[:, :, txt_length:, :]
            value = value[:, :, txt_length:, :]
        # =============================================================================

        # 应用自适应实例归一化进行风格对齐
        query = adain(query)
        key = adain(key)

        # =============================================================================
        # 重新拼接编码器状态和图像状态
        if encoder_hidden_states is None:
            query = torch.cat([encoder_hidden_states_q, query], dim=-2)
            key = torch.cat([encoder_hidden_states_k, key], dim=-2)
            value = torch.cat([encoder_hidden_states_v, value], dim=-2)
        # =============================================================================

        # FluxSingleTransformerBlock中的注意力不使用encoder_hidden_states
        if encoder_hidden_states is not None:
            # 编码器状态投影
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            # 编码器投影的归一化
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # 拼接查询、键、值
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        # =============================================================================
        # 处理旋转位置编码和特征拼接
        if txt_length is None:
            assert encoder_hidden_states is not None
            txt_length = encoder_hidden_states.shape[1]

        # 提取图像部分的键值并应用额外的旋转位置编码
        k_ = key[:, :, txt_length:, :]
        v_ = value[:, :, txt_length:, :]
        k_ = apply_rotary_emb(k_, image_rotary_emb_additional)

        # 应用标准旋转位置编码
        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

        # 将第一个样本的图像特征拼接到所有样本上
        key = concat_first_block(key, k_, -2, scale=self.scale)
        value = concat_first_block(value, v_, -2)
        # =============================================================================

        # =============================================================================
        # 计算注意力，可选择约束第一个样本
        if self.args.constrain_first:
            rows, cols = query.shape[-2], key.shape[-2]
            # 创建注意力掩码，限制第一个样本不关注额外添加的风格信息
            attn_mask = torch.zeros((query.shape[0], 24, rows, cols), dtype=query.dtype, device=query.device)
            attn_mask[0, :, :, query.shape[-2]:] = -float("inf")
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False,
                                                           attn_mask=attn_mask)
            del attn_mask
        else:
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        # =============================================================================

        # 重塑隐藏状态
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # 如果有编码器状态，分离处理
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1]:],
            )

            # 线性投影
            hidden_states = attn.to_out[0](hidden_states)
            # Dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
