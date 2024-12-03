import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import jax.numpy as jnp
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# 定义激活函数
ACTIVATION = {
    'gelu': nn.GELU(),
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(0.1),
    'silu': nn.SiLU()
}

# 位置编码模块
class PositionalEncoding:
    @staticmethod
    def generate_1d_encoding(embed_dim, positions):
        assert embed_dim % 2 == 0
        freqs = torch.arange(embed_dim // 2, dtype=torch.float32)
        freqs /= embed_dim / 2.0
        freqs = 1.0 / 10000**freqs  # (embed_dim/2,)

        positions = positions.reshape(-1)  # (N,)
        angles = torch.einsum("n,d->nd", positions, freqs)  # (N, embed_dim/2)

        sin_part = torch.sin(angles)  # (N, embed_dim/2)
        cos_part = torch.cos(angles)  # (N, embed_dim/2)

        return torch.cat([sin_part, cos_part], dim=1)  # (N, embed_dim)

    @staticmethod
    def generate_nd_encoding(embed_dim, grid_shape):
        assert embed_dim % len(grid_shape) == 0, \
            "Embedding dimension must be divisible by the number of dimensions."
        dim_per_axis = embed_dim // len(grid_shape)  # 每个维度的嵌入维度

        grids = [torch.arange(size, dtype=torch.float32) for size in grid_shape]
        mesh = torch.meshgrid(*grids, indexing="ij")  # 创建网格
        flat_mesh = torch.stack([g.reshape(-1) for g in mesh], dim=0)  # (D, prod(grid_shape))

        # 对每个维度进行位置编码，并沿最后一维拼接
        encodings = [
            PositionalEncoding.generate_1d_encoding(dim_per_axis, flat_mesh[dim])
            for dim in range(len(grid_shape))
        ]
        combined_encoding = torch.cat(encodings, dim=1)  # (prod(grid_shape), embed_dim)

        return combined_encoding.unsqueeze(0)  # (1, prod(grid_shape), embed_dim)


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rates=None, activation='relu', stride=1):

        if dilation_rates is None:
            dilation_rates = [1, 2, 3, 5]

        super(MultiScaleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates
        self.activation_name = activation
        self.activation = self.get_activation(activation)
        self.stride = stride

        # 初始化卷积层列表
        self.conv_layers = nn.ModuleList()

        for rate in dilation_rates:
            # 根据膨胀率计算适当的填充，以确保感受野完整覆盖
            padding = tuple(((k - 1) * rate) // 2 for k in self._to_tuple(kernel_size))
            Conv = getattr(nn, f'Conv{len(padding)}d')
            self.conv_layers.append(
                Conv(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self._to_tuple(kernel_size),
                    dilation=rate,
                    stride=self._to_tuple(stride),
                    padding=padding,
                    bias=False
                )
            )

    def get_activation(self, activation):
        activation_map = {
            'relu': nn.ReLU(inplace=True),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(inplace=True),
        }
        return activation_map.get(activation, nn.ReLU(inplace=True))

    def forward(self, x):
        outputs = []
        for conv in self.conv_layers:
            out = conv(x)
            out = self.activation(out)
            outputs.append(out)
        # 在通道维度上拼接输出
        return torch.cat(outputs, dim=1)

    def _to_tuple(self, value):
        # 将输入转换为元组，以适应任意维度
        if isinstance(value, int):
            return (value,) * (len(self.kernel_size) if hasattr(self.kernel_size, '__len__') else 1)
        return value

class PatchExtractor(nn.Module):
    def __init__(self, patch_size, input_channels, embedding_dim, output_dim, activation='gelu', dilation_rates=[1, 2, 3, 5]):
        super().__init__()

        self.patch_size = patch_size
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.dilation_rates = dilation_rates
        self.activation_name = activation
        self.activation = self.get_activation(activation)
        self.projection = None  # 将在 forward 中初始化

    def get_activation(self, activation):
        activation_map = {
            'relu': nn.ReLU(inplace=True),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(inplace=True),
        }
        return activation_map.get(activation, nn.ReLU(inplace=True))

    def forward(self, x):
        """
        输入张量 x 形状为 (B, C, D1, D2, ..., DN)
        输出张量形状为 (B, output_dim, G1, G2, ..., GN)
        """
        B, C, *dims = x.shape
        dims_num = len(dims)

        # 将 patch_size 转换为元组
        if isinstance(self.patch_size, int):
            patch_size = (self.patch_size,) * dims_num
        else:
            patch_size = self.patch_size

        # 根据维度选择合适的卷积层
        Conv = getattr(nn, f'Conv{dims_num}d')

        if self.projection is None:
            self.projection = nn.Sequential(
                MultiScaleConv(
                    in_channels=self.input_channels,
                    out_channels=self.embedding_dim,
                    kernel_size=patch_size,
                    dilation_rates=self.dilation_rates,
                    activation=self.activation_name,
                    stride=patch_size  # 使用 patch_size 作为步幅
                ),
                self.activation,
                Conv(
                    in_channels=len(self.dilation_rates) * self.embedding_dim,
                    out_channels=self.output_dim,
                    kernel_size=1,
                    stride=1
                )
            )

        patches = self.projection(x)
        return patches




class Mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, activation='gelu', dropout_rate=0.0):
        super().__init__()
        output_dim = output_dim or input_dim
        hidden_dim = hidden_dim or input_dim
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.activation = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(inplace=True),
            'silu': nn.SiLU(inplace=True),
        }[activation]
        self.dense2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        """
        输入张量形状为 (B, N, input_dim)
        输出张量形状为 (B, N, output_dim)
        """
        x = self.dense1(x)           #  (B, N, hidden_dim)
        x = self.activation(x)       
        x = self.dropout(x)          
        x = self.dense2(x)           #  (B, N, output_dim)
        return x


class ConvResFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, truncation_size=16):
        """
        param truncation_size: 傅里叶变换时保留的低频模式数
        """
        super(ConvResFourierLayer, self).__init__()
        
        # 卷积层：捕获高频特征
        self.dims = None  # 在 forward 中根据输入数据确定维度
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # 偏置项
        self.bias = nn.Parameter(torch.zeros(1))
        
        # 傅里叶域中的核的截断大小
        self.truncation_size = truncation_size

    def fourier_transform(self, x):
        """
        将输入x从空间域转换到频域
        """
        return torch.fft.fftn(x, dim=tuple(range(2, x.dim())))

    def inverse_fourier_transform(self, x):
        """
        将频域表示转换回空间域
        """
        return torch.fft.ifftn(x, dim=tuple(range(2, x.dim()))).real

    def truncate_low_frequencies(self, x_fft):
        """
        截断傅里叶频谱的高频部分，仅保留低频信息
        param x_fft:频域表示的张量
        返回截断后的频域表示
        """
        # 获取频域的空间维度数，先排除 batch 和通道维度
        spatial_dims = x_fft.dim() - 2
        
        # 为每个空间维度生成切片
        slices = [slice(None)] * 2 + [slice(0, self.truncation_size)] * spatial_dims
        
        # 使用切片截取低频部分
        return x_fft[tuple(slices)]

    def forward(self, x):
        if self.dims is None:
            self.dims = x.dim() - 2  # 计算空间维度
            Conv = getattr(nn, f'Conv{self.dims}d')
            self.conv = Conv(
                self.in_channels, self.out_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2
            )

        # 获取输入的频域表示
        x_fft = self.fourier_transform(x)
        
        # 保留低频部分
        x_fft = self.truncate_low_frequencies(x_fft)
        
        # 获取卷积核的频域表示
        weight = self.conv.weight
        # 对卷积核进行 zero-padding，使其尺寸与输入匹配
        pad_sizes = []
        for dim_size in x.shape[2:]:
            pad_sizes.extend([0, dim_size - weight.shape[2]])
        weight_padded = F.pad(weight, pad_sizes[::-1])

        kernel_fft = torch.fft.fftn(weight_padded, dim=tuple(range(2, weight_padded.dim())))
        kernel_fft = self.truncate_low_frequencies(kernel_fft)

        # 在傅里叶域中进行卷积
        out_fft = x_fft * kernel_fft

        # 反傅里叶变换
        out = self.inverse_fourier_transform(out_fft)

        # 卷积层处理高频部分
        conv_out = self.conv(x)

        # 残差连接
        out = out + conv_out + self.bias

        return out



class SelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, mlp_ratio, layer_norm_eps=1e-5, dropout_rate=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps

        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.mlp_dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        """
        inputs (torch.Tensor)输入张量，形状为 (B, *Spatial, D)。
        输出张量形状与输入相同。
        """
        B, *spatial_dims, D = inputs.shape
        assert D == self.embed_dim, f"Input embedding dimension {D} must match {self.embed_dim}."

        # 将空间维度合并为序列维度
        x = rearrange(inputs, "b ... d -> b (...) d")  # (B, S, D)，S 为展开的空间维度

        # 多头注意力
        x_residual = x
        x = self.norm1(x)  # 层归一化
        x, _ = self.attention(x, x, x)  # (B, S, D)
        x = self.attention_dropout(x)  # Dropout 后的注意力输出
        x = x + x_residual  # 残差连接

        # MLP
        y_residual = x
        x = self.norm2(x)  # 层归一化
        x = self.mlp(x)  # (B, S, D)
        x = self.mlp_dropout(x)  # Dropout 后的 MLP 输出
        x = x + y_residual  # 残差连接

        # 恢复原始空间维度
        pattern = "b s d -> b " + " ".join([f"d{i}" for i in range(len(spatial_dims))]) + " d"
        x = rearrange(x, pattern, s=torch.prod(torch.tensor(spatial_dims)).item(), **{f"d{i}": spatial_dims[i] for i in range(len(spatial_dims))})
        return x




class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, layer_norm_eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.mlp = Mlp(
            input_dim=embed_dim,
            hidden_dim=int(embed_dim * mlp_ratio),
            output_dim=embed_dim,
            activation='gelu'
        )

    def forward(self, latents, x):
        """
        latents(torch.Tensor)隐变量，形状为(B, S, T', D)
        x(torch.Tensor)输入特征，形状为(B, S, T, D)

        返回更新后的隐变量，形状为(B, S, T', D)
        """
        B, S, T_prime, D = latents.shape
        _, _, T, _ = x.shape

        # 调整维度以匹配 MultiheadAttention 的输入要求
        latents_flat = rearrange(latents, "b s t d -> (b s) t d")  # (B*S, T', D)
        x_flat = rearrange(x, "b s t d -> (b s) t d")  # (B*S, T, D)

        # 隐变量和输入特征之间的注意力
        latents_attn = self.cross_attention(
            self.norm1(latents_flat), self.norm1(x_flat), self.norm1(x_flat)
        )[0]  # (B*S, T', D)

        # 残差连接
        latents_flat = latents_flat + latents_attn

        # MLP 层
        latents_flat = latents_flat + self.mlp(self.norm2(latents_flat))

        # 恢复形状
        latents = rearrange(latents_flat, "(b s) t d -> b s t d", b=B, s=S)
        return latents


        # 时间聚合模块
class TimeAggregator(nn.Module):
    def __init__(self, embed_dim, depth, num_heads=8, num_latents=64, mlp_ratio=1, layer_norm_eps=1e-5):
        """
        num_latents (int): 时间聚合的隐变量数量 T'
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_latents = num_latents
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps

        # 初始化可学习的时间隐变量
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))  # (T', D)

        # Transformer 层
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, layer_norm_eps)
            for _ in range(depth)
        ])

    def forward(self, x):
        """
        参数x(torch.Tensor)输入张量，形状为(B, T, *S_dims, D)
        返回torch.Tensor输出张量，形状为(B, T', *S_dims, D)。
        """
        B, T, *S_dims, D = x.shape
        assert D == self.embed_dim, f"Input embedding dimension {D} must match {self.embed_dim}."

        # 合并空间维度为单一维度
        S = torch.prod(torch.tensor(S_dims)).item()
        x_flat = rearrange(x, "b t ... d -> b t s d", s=S)  # (B, T, S, D)

        # 初始化隐变量，扩展为每个批次和空间维度
        latents = repeat(self.latents, "t d -> b s t d", b=B, s=S)  # (B, S, T', D)

        # 调整输入形状为 (B, S, T, D)
        x_flat = rearrange(x_flat, "b t s d -> b s t d")

        # Transformer 层聚合
        for block in self.cross_attn_blocks:
            latents = block(latents, x_flat)

        # 调整输出形状为 (B, T', *S_dims, D)，动态解包 S_dims
        pattern = "b s t d -> b t " + " ".join([f"d{i}" for i in range(len(S_dims))])
        latents = rearrange(latents, pattern, s=S, **{f"d{i}": S_dims[i] for i in range(len(S_dims))})
        return latents


def F_avg_pool_nd(x, kernel_size):
    dims = x.dim() - 2  # 减去 batch 和 channel 维度
    pool = getattr(F, f'avg_pool{dims}d')
    return pool(x, kernel_size)


def generate_rearrange_pattern(spatial_shape, flatten=False):
    """
    根据输入的空间维度生成动态的rearrange模式。
    :param spatial_shape: 空间维度列表 (D1, D2, ..., Dn)
    :param flatten: 是否展平空间维度
    :return: rearrange模式字符串
    """
    dims = ' '.join([f'd{i}' for i in range(len(spatial_shape))])
    if flatten:
        return f'b c {dims} -> b ({dims}) c'
    else:
        return f'b s c -> b c {dims}'



class HierarchicalFNO(nn.Module):
    def __init__(self, img_size=(64, 64), patch_size=4, in_channels=1, out_channels=1, embed_dim=64, depth=4, num_scales=3, truncation_sizes=None, num_heads=4, mlp_ratio=2.0, activation='gelu'):
        super(HierarchicalFNO, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_scales = num_scales
        self.activation_name = activation
        self.activation = ACTIVATION[activation]
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # 空间维度数量
        self.spatial_dims = len(img_size)

        # Patch提取器
        self.patch_extractor = PatchExtractor(
            patch_size=patch_size,
            input_channels=in_channels,
            embedding_dim=embed_dim,
            output_dim=embed_dim,
            activation=activation,
            dilation_rates=[1, 2, 3, 5]
        )

        # 分层的卷积-残差傅里叶层
        if truncation_sizes is None:
            truncation_sizes = [16, 12, 8]
        assert len(truncation_sizes) == num_scales, "截断大小的数量应与尺度数量相匹配。"

        self.conv_res_fourier_layers = nn.ModuleList()
        for scale in range(num_scales):
            layer = nn.ModuleList([
                ConvResFourierLayer(
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    kernel_size=3,
                    truncation_size=truncation_sizes[scale]
                )
                for _ in range(depth)
            ])
            self.conv_res_fourier_layers.append(layer)

        # 自注意力层
        self.self_attention_layers = nn.ModuleList([
            SelfAttention(
                num_heads=num_heads,
                embed_dim=embed_dim,
                mlp_ratio=mlp_ratio,
                layer_norm_eps=1e-5,
                dropout_rate=0.1
            )
            for _ in range(num_scales)
        ])

        # 输出MLP层
        self.mlp_head = Mlp(
            input_dim=embed_dim,
            hidden_dim=int(embed_dim * mlp_ratio),
            output_dim=out_channels,
            activation=activation
        )

    def forward(self, x):
        B, C, *spatial_shape = x.shape
        spatial_dims = len(spatial_shape)
        assert spatial_dims == self.spatial_dims, f"输入的空间维度数量应为 {self.spatial_dims}，但得到的是 {spatial_dims}"

        # Patch提取
        x = self.patch_extractor(x)  # (B, embed_dim, *new_spatial_shape)
        new_spatial_shape = x.shape[2:]  # 更新后的空间维度

        # 在此生成位置编码
        pos_encoding = PositionalEncoding.generate_nd_encoding(self.embed_dim, new_spatial_shape)
        pos_encoding = torch.tensor(pos_encoding, dtype=torch.float32).to(x.device)
        pos_encoding = pos_encoding.permute(0, *range(2, 2 + spatial_dims), 1)  # 调整形状为 (1, *new_spatial_shape, embed_dim)
        x = x.permute(0, *range(2, 2 + spatial_dims), 1)  # 调整形状为 (B, *new_spatial_shape, embed_dim)
        x = x + pos_encoding  # 形状匹配，相加
        x = x.permute(0, -1, *range(1, spatial_dims + 1))  # 恢复形状为 (B, embed_dim, *new_spatial_shape)



        # 分层处理
        outputs = []
        for scale_idx in range(self.num_scales):
            # 对更高尺度的输入进行下采样（可选）
            if scale_idx > 0:
                x = F.avg_pool_nd(x, kernel_size=2)

            # 通过卷积-残差傅里叶层
            for layer in self.conv_res_fourier_layers[scale_idx]:
                x = layer(x)

            # 自注意力层
            pattern_flatten = generate_rearrange_pattern(spatial_shape, flatten=True)
            x = rearrange(x, pattern_flatten)
            x = self.self_attention_layers[scale_idx](x)
            pattern_restore = generate_rearrange_pattern(spatial_shape, flatten=False)
            x = rearrange(x, pattern_restore, **{f'd{i}': spatial_shape[i] for i in range(len(spatial_shape))})


            # 上采样并存储输出
            if scale_idx > 0:
                scale_factor = 2 ** scale_idx
                x = F.interpolate(x, scale_factor=scale_factor, mode='trilinear' if spatial_dims == 3 else 'bilinear', align_corners=False)
            outputs.append(x)


        # 多尺度输出的形状一致
        # for i, out in enumerate(outputs):
        #     if out.shape != outputs[0].shape:
        #         outputs[i] = F.interpolate(out, size=outputs[0].shape[2:], mode='trilinear' if spatial_dims == 3 else 'bilinear', align_corners=False)


        # 合并不同尺度的输出
        x = torch.stack(outputs, dim=0).sum(dim=0)  

       # 最后的MLP层
        pattern_flatten = generate_rearrange_pattern(spatial_shape, flatten=True)
        x = rearrange(x, pattern_flatten)
        x = self.mlp_head(x)
        pattern_restore = generate_rearrange_pattern(spatial_shape, flatten=False)
        x = rearrange(x, pattern_restore, **{f'd{i}': spatial_shape[i] for i in range(len(spatial_shape))})


        return x



# 示例用法
if __name__ == "__main__":
    # 创建随机输入张量，例如3D数据
    x = torch.randn(2, 1, 32, 32, 32)  # (B, C, D, H, W)

    # 实例化模型，指定输入维度
    model = HierarchicalFNO(
        img_size=(32, 32, 32),
        patch_size=4,
        in_channels=1,
        out_channels=1,
        embed_dim=64,
        depth=2,
        num_scales=3,
        truncation_sizes=[16, 12, 8],
        num_heads=4,
        mlp_ratio=2.0,
        activation='gelu'
    )

    # 前向传播
    output = model(x)
    print(f"输出形状: {output.shape}")  # 应该是 (B, out_channels, D, H, W)

