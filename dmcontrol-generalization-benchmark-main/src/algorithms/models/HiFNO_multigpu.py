import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange



ACTIVATION = {
    'gelu': nn.GELU(),
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(0.1),
    'silu': nn.SiLU()
}


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
        mesh = torch.meshgrid(*grids, indexing="ij")  
        flat_mesh = torch.stack([g.reshape(-1) for g in mesh], dim=0)  # (D, prod(grid_shape))

        # 每个维度进行位置编码，沿最后一维拼接
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

        # 将kernel_size和stride转换为元组
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        
        if isinstance(stride, int):
            self.stride = (stride, stride)
        
        self.conv_layers = nn.ModuleList()

        for rate in dilation_rates:
            padding = tuple(((k - 1) * rate) // 2 for k in self._to_tuple(kernel_size))
            conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self._to_tuple(kernel_size),
                dilation=rate,
                stride=self._to_tuple(stride),
                padding=padding,
                bias=False
            )
            self.conv_layers.append(conv)

    def get_activation(self, activation):
        activation_map = {
            'relu': nn.ReLU(inplace=True),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(inplace=True),
        }
        return activation_map.get(activation, nn.ReLU(inplace=True))

    def forward(self, x):
        outputs = []
        expected_size = None
        
        for i, conv in enumerate(self.conv_layers):
            out = conv(x)
            out = self.activation(out)
            
            # 记录第一个输出的大小作为基准
            if i == 0:
                expected_size = out.shape[2:]
            else:
                # 检查所有输出大小一致
                current_size = out.shape[2:]
                if current_size != expected_size:
                    out = F.interpolate(out, size=expected_size, mode='bilinear', align_corners=False)
            
            outputs.append(out)
        
        
        return torch.cat(outputs, dim=1)

    def _to_tuple(self, value):
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
        self.projection = None  

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
        
        
        if self.projection is None or self.projection[0].in_channels != C:
            # 每个分支的输出通道数
            branch_channels = self.embedding_dim

            # 新的投影层
            new_proj = nn.Sequential(
                MultiScaleConv(
                    in_channels=C,  # 用实际的输入通道数
                    out_channels=branch_channels,
                    kernel_size=self.patch_size,
                    dilation_rates=self.dilation_rates,
                    activation=self.activation_name,
                    stride=self.patch_size
                ),
                self.activation,
                nn.Conv2d(  
                    in_channels=branch_channels * len(self.dilation_rates),
                    out_channels=self.output_dim,
                    kernel_size=1,
                    stride=1
                )
            )
            
            
            new_proj = new_proj.to(x.device)
            self.projection = new_proj

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
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.truncation_size = truncation_size
        
        
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
        
        self.scale = 0.02
        self.weights = nn.Parameter(self.scale * torch.randn(2, in_channels, truncation_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def _update_conv(self, x):
        dims = x.dim() - 2
        # 获取当前卷积层所在的设备
        old_device = self.conv.weight.device
        
        if dims != len(self.conv.weight.shape) - 2:
            Conv = getattr(nn, f'Conv{dims}d')
            new_conv = Conv(
                self.in_channels, 
                self.out_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2
            )
            
            new_conv = new_conv.to(old_device)
            
            # 复制权重和偏置
            if hasattr(self.conv, 'weight'):
                # 这里可能需要具体情况调整权重的复制方式
                new_conv.weight.data[:] = self.conv.weight.data.unsqueeze(-1) if dims > 1 else self.conv.weight.data
            if hasattr(self.conv, 'bias') and self.conv.bias is not None:
                new_conv.bias.data[:] = self.conv.bias.data
            
            # 更新卷积层引用
            self.conv = new_conv

    def forward(self, x):
        self._update_conv(x)
        conv_out = self.conv(x)
        
        
        x_fft = torch.fft.fftn(x, dim=tuple(range(2, x.dim())))
        x_fft = x_fft[..., :self.truncation_size, :self.truncation_size]

        pad_sizes = []
        for dim_size in x.shape[2:]:
            pad_sizes.extend([0, dim_size - self.conv.weight.shape[2]])
        weight_padded = F.pad(self.conv.weight, pad_sizes[::-1])

        
        kernel_fft = torch.fft.fftn(weight_padded, dim=tuple(range(2, weight_padded.dim())))
        kernel_fft = kernel_fft[..., :self.truncation_size, :self.truncation_size]

        # 调整维度以进行批量卷积
        B = x_fft.size(0)
        kernel_fft = kernel_fft.unsqueeze(0).expand(B, -1, -1, -1, -1)

        
        out_fft = torch.einsum('bcxy,bkcxy->bkxy', x_fft, kernel_fft)

        
        out = torch.fft.ifftn(out_fft, s=x.shape[2:], dim=tuple(range(2, out_fft.dim())))
        out = out.real

        return out + conv_out + self.bias


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

    def forward(self, x):
        x_residual = x
        
        # 归一化和多头注意力
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.attention_dropout(attn_output)
        x = attn_output + x_residual

        # 归一化和 MLP
        y_residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.mlp_dropout(x)
        x = x + y_residual

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
        B, S, T_prime, D = latents.shape
        _, _, T, _ = x.shape

        
        latents_flat = rearrange(latents, "b s t d -> (b s) t d")  # (B*S, T', D)
        x_flat = rearrange(x, "b s t d -> (b s) t d")  # (B*S, T, D)

        # 隐量和输入特征之间的注意力
        latents_attn = self.cross_attention(
            self.norm1(latents_flat), self.norm1(x_flat), self.norm1(x_flat)
        )[0]  # (B*S, T', D)

        
        latents_flat = latents_flat + latents_attn

        
        latents_flat = latents_flat + self.mlp(self.norm2(latents_flat))

        
        latents = rearrange(latents_flat, "(b s) t d -> b s t d", b=B, s=S)
        return latents


        
class TimeAggregator(nn.Module):
    def __init__(self, embed_dim, depth, num_heads=8, num_latents=64, mlp_ratio=1, layer_norm_eps=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_latents = num_latents
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps

        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))

        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, layer_norm_eps)
            for _ in range(depth)
        ])

        self.dim_adjust = None

    def _create_dim_adjust(self, in_dim):
        """动态创建维度调整层"""
        self.dim_adjust = nn.Sequential(
            nn.Linear(in_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # 内部参数与 x 处于同一 device
        device = x.device
        self.latents = self.latents.to(device)
        if self.dim_adjust is not None:
            self.dim_adjust = self.dim_adjust.to(device)

        B = x.shape[0]
        orig_shape = x.shape
        D = orig_shape[-1]
        spatial_dims = orig_shape[1:-1]  # 获取所有中间维度

        S = np.prod(spatial_dims) if spatial_dims else 1
        x = x.view(B, -1, D)  # (B, S, D)

        # 添加时间维度（如果没有）
        x = x.unsqueeze(1)  # (B, 1, S, D)
        T = x.shape[1]

        # 动态创建维度调整层
        if D != self.embed_dim:
            if self.dim_adjust is None or self.dim_adjust[0].in_features != D:
                self._create_dim_adjust(D)
            self.dim_adjust = self.dim_adjust.to(device)

        x = x.view(-1, D)
        x = self.dim_adjust(x).view(B, T, S, self.embed_dim)

        # 初始化隐变量扩展为每个批次和空间维度
        latents = repeat(self.latents, "t d -> b s t d", b=B, s=S)

        x = rearrange(x, "b t s d -> b s t d")

        for block in self.cross_attn_blocks:
            latents = block(latents, x)

        # 恢复原始空间维度
        if spatial_dims:
            latents = latents.view(B, *spatial_dims, self.num_latents, self.embed_dim)
            # 调整维度顺序，将时间维度移到第二
            perm = [0, len(spatial_dims) + 1] + list(range(1, len(spatial_dims) + 1)) + [len(spatial_dims) + 2]
            latents = latents.permute(*perm)  # (B, T', *spatial_dims, embed_dim)

        return latents


def F_avg_pool_nd(x, kernel_size):
    dims = x.dim() - 2  # 减去 batch 和 channel 维度
    pool = getattr(F, f'avg_pool{dims}d')
    return pool(x, kernel_size)


def generate_rearrange_pattern(spatial_shape, flatten=False):
    dims = ' '.join([f'd{i}' for i in range(len(spatial_shape))])
    if flatten:
        # 从 (B, C, D1, D2, ..., Dn) 到 (B, N, C)
        return f'b c {dims} -> b ({dims}) c'
    else:
        # 从 (B, N, C) 到 (B, C, D1, D2, ..., Dn)
        return f'b n c -> b c {dims}'


def get_interpolation_mode(spatial_dims):
    modes = {
        1: 'linear',
        2: 'bilinear',
        3: 'trilinear'
    }
    return modes.get(spatial_dims, 'trilinear')


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

        
        self.patch_extractor = PatchExtractor(
            patch_size=patch_size,
            input_channels=in_channels,
            embedding_dim=embed_dim,
            output_dim=embed_dim,
            activation=activation,
            dilation_rates=[1, 2, 3, 5]
        )

        
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

        
        self.mlp_head = Mlp(
            input_dim=embed_dim,
            hidden_dim=int(embed_dim * mlp_ratio),
            output_dim=out_channels,
            activation=activation
        )

    def forward(self, x):
        B, C, *spatial_shape = x.shape
        spatial_dims = len(spatial_shape)
        assert spatial_dims == self.spatial_dims, f"入的空间维度数量应为 {self.spatial_dims}，但得到的是 {spatial_dims}"

        # 特征提取
        x = self.patch_extractor(x)
        new_spatial_shape = x.shape[2:]
        base_shape = new_spatial_shape

        # 位置编码
        pos_encoding = PositionalEncoding.generate_nd_encoding(
            self.embed_dim, 
            new_spatial_shape
        ).to(x.device)
        x = rearrange(x, 'b c ... -> b ... c')
        pos_encoding = pos_encoding.view(1, *new_spatial_shape, -1)
        x = x + pos_encoding
        x = rearrange(x, 'b ... c -> b c ...')

        # 分层处理
        outputs = []
        for scale_idx in range(self.num_scales):
            scale_x = x
            
            if scale_idx > 0:
                scale_x = F_avg_pool_nd(scale_x, kernel_size=2)
            
            for layer in self.conv_res_fourier_layers[scale_idx]:
                scale_x = layer(scale_x)
            
            current_shape = scale_x.shape[2:]
            scale_x = rearrange(scale_x, 'b c ... -> b (...) c')
            scale_x = self.self_attention_layers[scale_idx](scale_x)
            
            scale_x = rearrange(
                scale_x, 
                f'b ({" ".join(f"d{i}" for i in range(len(current_shape)))}) c -> b c {" ".join(f"d{i}" for i in range(len(current_shape)))}',
                **{f'd{i}': current_shape[i] for i in range(len(current_shape))}
            )

            if scale_x.shape[2:] != base_shape:
                scale_x = F.interpolate(scale_x, size=base_shape, mode='bilinear', align_corners=False)
            
            outputs.append(scale_x)

        x = torch.stack(outputs, dim=0).sum(dim=0)
        
        # 先将 x 从 (B, C, D1, D2, ...) 重排到 (B, N, C)
        x = rearrange(x, 'b c ... -> b (...) c')  
        
        # 对 N 维度做平均池化，得到特征向量
        x = x.mean(dim=1)  # [B, C]
        
        # 通过 MLP 得到最终特征
        x = self.mlp_head(x)  # [B, out_channels]
        
        # 重塑为 4D 张量
        H = W = int(np.sqrt(self.out_channels))
        x = x.view(B, H, W, 1)
        x = x.permute(0, 3, 1, 2)  # [B, 1, H, W]
        
        return x




if __name__ == "__main__":
    
    x = torch.randn(2, 1, 32, 32, 32)  # (B, C, D, H, W)

    
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

