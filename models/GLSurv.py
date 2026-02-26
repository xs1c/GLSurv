from __future__ import annotations
import torch.nn as nn
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
import torch.nn.functional as F
import torch
from einops import repeat
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

class MambaLayer(nn.Module):
    """
    3D 版本 MambaLayer（按新思想修改）：
    - 输入: x (B, D, H, W, C=d_model)
    - 步骤:
        1) 3D 拉普拉斯频域分解 -> 低频 x_low, 高频 x_high (均为B,D,H,W,C)
        2) 高低频在通道维拼接 -> (B,D,H,W,2C)
        3) 3D卷积将通道数复原为C (d_model)，得到频率融合特征 x_f
        4) CSS3D(x, x_f) 捕捉原始特征-频率特征的互补信息，作为最终输出
    """

    def __init__(
            self,
            d_model: int,
            lap_num_iters: int = 10,
            lap_time_step: float = 0.1,
            # 下面这些参数直接透传给 CSS3D
            d_state=16,
            d_conv=3,
            ssm_ratio=2,
            dt_rank="auto",
            dropout=0.1,
            conv_bias=True,
            bias=False,
            dtype=None,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            shared_ssm=False,
            softmax_version=False,
            **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.lap_num_iters = lap_num_iters
        self.lap_time_step = lap_time_step

        factory_kwargs = {"device": None, "dtype": dtype}

        base_kernel = torch.zeros((3, 3, 3), dtype=torch.float32)
        base_kernel[1, 1, 1] = -6.0
        base_kernel[0, 1, 1] = 1.0  # z-1
        base_kernel[2, 1, 1] = 1.0  # z+1
        base_kernel[1, 0, 1] = 1.0  # y-1
        base_kernel[1, 2, 1] = 1.0  # y+1
        base_kernel[1, 1, 0] = 1.0  # x-1
        base_kernel[1, 1, 2] = 1.0  # x+1
        base_kernel = base_kernel.view(1, 1, 3, 3, 3)  # (1,1,3,3,3)

        # 为每个通道复制一份: (C,1,3,3,3)，用于 depthwise conv3d
        lap_kernel = base_kernel.repeat(d_model, 1, 1, 1, 1)
        self.register_buffer("lap_kernel", lap_kernel.to(**factory_kwargs), persistent=False)

        # 拼接后通道数=2*d_model，需卷积还原为d_model
        # Conv3D输入格式：(B, C_in, D, H, W)，因此in_channels=2*d_model, out_channels=d_model
        self.concat_conv = nn.Conv3d(
            in_channels=2 * d_model,
            out_channels=d_model,
            kernel_size=1,  # 3x3x3卷积捕捉局部频率融合信息
            padding=0,  # 保持D/H/W维度不变
            bias=conv_bias,
            **factory_kwargs
        )

        self.drop=nn.Dropout(0.1)
        self.fcss3d = CSS3D(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            dropout=dropout,
            conv_bias=conv_bias,
            bias=bias,
            dtype=dtype,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs,
        )

        self.spatial_pool_conv = nn.Conv3d(
            in_channels=2 * d_model,
            out_channels=d_model,
            kernel_size=1,
            padding=0,
            bias=conv_bias,
            **factory_kwargs
        )

    def laplacian_smoothing_3d(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用 3D 拉普拉斯算子进行迭代平滑:
            u_{k+1} = u_k + tau * Δu_k

        x: (B, D, H, W, C)
        返回: 同形状 (B, D, H, W, C)
        """
        assert x.dim() == 5, "输入必须是 [B, D, H, W, C]"
        u = x

        for _ in range(self.lap_num_iters):
            # 转换为Conv3D输入格式: (B, C, D, H, W)
            u_perm = u.permute(0, 4, 1, 2, 3).contiguous()
            lap_u = F.conv3d(
                u_perm,
                self.lap_kernel.to(u.dtype).to(u.device),
                padding=1,
                groups=self.d_model,  # depthwise
            )
            # 转换回原格式: (B, D, H, W, C)
            lap_u = lap_u.permute(0, 2, 3, 4, 1).contiguous()
            u = u + self.lap_time_step * lap_u

        return u

    def laplacian_decomposition_3d(self, x: torch.Tensor):
        """
        频域分解:
            low = 拉普拉斯平滑(x)
            high = x - low
        x: (B, D, H, W, C)
        返回: (low, high), 形状同 x
        """
        low = self.laplacian_smoothing_3d(x)
        high = x - low
        return low, high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1=x
        xb = x.permute(0, 2, 3, 4, 1).contiguous()
        x_low, x_high = self.laplacian_decomposition_3d(xb)

        # 2. 高低频在通道维拼接 -> (B,D,H,W,2C)
        x_concat = torch.cat([x_low, x_high], dim=-1)

        # 3. 转换为Conv3D输入格式: (B, 2C, D, H, W)
        x_concat_perm = x_concat.permute(0, 4, 1, 2, 3).contiguous()
        x_concat_perm = self.drop(x_concat_perm)
        # 4. 3D卷积还原通道数为C -> (B, C, D, H, W)
        x_f = self.concat_conv(x_concat_perm)

        x_max = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)  # 局部最大值池化

        x_avg = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)  # 局部平均值池化

        # 步骤3: 通道维度拼接 (B, 2C, D, H, W)
        x_pool_concat = torch.cat([x_max, x_avg], dim=1)

        # 步骤4: 1x1x1卷积还原通道数为d_model (B, C, D, H, W)
        x_pool = self.spatial_pool_conv(x_pool_concat)
        # 步骤5: 转换回原格式 (B, D, H, W, C)，适配CSS3D输入
        # ===== 修改：将增强后的x_pool传入CSS3D =====
        out = self.fcss3d(x_pool, x_f)  # (B, D, H, W, d_model)
        # 残差连接 + 维度调整
        out = out.permute(0, 4, 1, 2, 3).contiguous()
        out = out + x1
        return out

class CSS3D(nn.Module):
    """3D版本交叉特征融合模块，基于SSM（状态空间模型），仅四向交互"""
    def __init__(
        self,
        d_model,               # 输入特征通道数（如1152）
        d_state=8,            # SSM状态维度
        d_conv=3,              # 3D卷积核大小
        ssm_ratio=2,           # 特征扩展比例（d_inner = d_model * ssm_ratio）
        dt_rank="auto",        # dt投影的秩
        dropout=0.1,            #  dropout概率
        conv_bias=True,        # 卷积层偏置
        bias=False,            # 线性层偏置
        dtype=None,            # 数据类型
        dt_min=0.001,          # dt最小值
        dt_max=0.1,            # dt最大值
        dt_init="random",      # dt初始化方式
        dt_scale=1.0,          # dt缩放系数
        dt_init_floor=1e-4,    # dt初始化下限
        shared_ssm=False,      # 是否共享SSM参数
        softmax_version=False, # 是否使用softmax版本
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": dtype}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        # 自动计算d_state（若未指定）
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)  # 扩展后的特征维度
        # 自动计算dt_rank（若未指定）
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.K = 4 if not shared_ssm else 1  # 四向交互，故K=4

        # 1. 输入投影层（主输入+交叉输入）
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_cross = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        # 2. 3D深度可分离卷积（提取局部空间特征）
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,  # 深度可分离：每个通道独立卷积
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,  # 保证卷积后空间尺寸不变
            **factory_kwargs,
        )
        self.act = nn.SiLU()  # 激活函数

        # 3. SSM参数生成层（四向独立参数）
        self.x_proj = [
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (4, dt_rank+2*d_state, d_inner)
        del self.x_proj  # 删除原列表，减少内存占用

        # 4. dt参数投影层（四向独立参数）
        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (4, d_inner, dt_rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))    # (4, d_inner)
        del self.dt_projs  # 删除原列表

        # 5. SSM核心状态参数（A_logs、Ds）
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True)  # (4*d_inner, d_state)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True)                        # (4*d_inner,)

        # 6. 输出归一化与投影
        if not self.softmax_version:
            self.out_norm = nn.LayerNorm(self.d_inner)  # 特征归一化
        # self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)  # 投影回原维度
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None  # 输出dropout

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        """初始化dt投影层，确保F.softplus(dt_bias)在[dt_min, dt_max]范围内"""
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        # 初始化权重：保持方差一致
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"不支持的dt初始化方式：{dt_init}")

        # 初始化偏置：用softplus逆运算确保初始dt在合理范围
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # softplus逆运算（参考PyTorch#72759）
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True  # 标记为无需重新初始化
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        """初始化SSM的A矩阵（取对数，保证数值稳定）"""
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",  # 每个通道对应一组1~d_state的序列
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # 取对数，后续用-exp(A_log)得到负的A矩阵（保证SSM稳定性）
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)  # 扩展为K=4组
            if merge:
                A_log = A_log.flatten(0, 1)  # 展平为(4*d_inner, d_state)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True  # 标记为无需权重衰减
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        """初始化SSM的跳跃连接参数D"""
        D = torch.ones(d_inner, device=device)  # 初始为1，保证跳跃连接有效
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)  # 扩展为K=4组
            if merge:
                D = D.flatten(0, 1)  # 展平为(4*d_inner,)
        D = nn.Parameter(D)
        D._no_weight_decay = True  # 标记为无需权重衰减
        return D

    def generate_3d_directions(self, feat):
        """生成3D特征的四向交互序列（正向、反向、通道反向、交叉扫描）"""
        B, C, D, H, W = feat.shape
        L = D * H * W  # 3D展平后的1D序列长度

        # 1. 正向序列：按D→H→W顺序展平（原始空间顺序）
        dir_forward = feat.view(B, C, L)

        # 2. 反向序列：正向序列首尾翻转（捕捉反向空间依赖）
        dir_backward = torch.flip(dir_forward, dims=[-1])

        # 3. 通道反向序列：先翻转通道维度，再展平（捕捉通道反向语义）
        feat_channel_flipped = torch.flip(feat, dims=[1])  # 翻转通道（dim=1）
        dir_channel_backward = feat_channel_flipped.view(B, C, L)

        # 4. 交叉扫描序列：首→尾→次首→次尾交替（捕捉跨区域长距离依赖）
        indices = torch.arange(L, device=feat.device)
        half = (L + 1) // 2
        front = indices[:half]
        back = indices[half:].flip(dims=[0])
        cross_indices = torch.stack([front, back], dim=1).flatten()[:L]  # 确保长度为L
        dir_cross = dir_forward[:, :, cross_indices]

        # 堆叠四向序列：(B, 4, C, L)
        return torch.stack([dir_forward, dir_backward, dir_channel_backward, dir_cross], dim=1)

    def forward_core(self, x: torch.Tensor, x_cross: torch.Tensor):
        """3D特征融合核心逻辑：四向交互+SSM长距离建模（按方向分配权重）"""
        self.selective_scan = selective_scan_fn
        B, C, D, H, W = x.shape
        L = D * H * W
        K = self.K  # 四向交互，K=4

        # 生成四向序列（主输入x + 交叉输入x_cross）
        xs = self.generate_3d_directions(x)  # (B, 4, C, L)
        xs_cross = self.generate_3d_directions(x_cross)  # (B, 4, C, L)

        # 准备SSM状态参数（按方向拆分，和2D版本对齐）
        # As/Ds/dt_projs_bias都是按4个方向拼接的，需拆分成单个方向的参数
        As = -torch.exp(self.A_logs.float())  # (4*C, d_state) → 拆成4个 (C, d_state)
        As_list = torch.split(As, C, dim=0)  # 按方向拆分As：[As_0, As_1, As_2, As_3]

        Ds = self.Ds.float()  # (4*C,) → 拆成4个 (C,)
        Ds_list = torch.split(Ds, C, dim=0)  # 按方向拆分Ds：[Ds_0, Ds_1, Ds_2, Ds_3]

        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (4*C,) → 拆成4个 (C,)
        dt_projs_bias_list = torch.split(dt_projs_bias, C, dim=0)  # 按方向拆分bias

        direction_outs = []
        for k in range(K):
            # ===================== 1. 取第k个方向的特征 =====================
            x_k = xs[:, k, :, :].float()  # (B, C, L)
            x_cross_k = xs_cross[:, k, :, :].float()  # (B, C, L)

            # ===================== 2. 取第k个方向的权重（核心！）=====================
            # ① 取第k个方向的x_proj_weight → 从(4,35,96) → (35,96)（2维，匹配einsum）
            x_proj_weight_k = self.x_proj_weight[k]
            # ② 取第k个方向的dt_projs_weight → 从(4,96,3) → (96,3)（2维，匹配dt投影）
            dt_projs_weight_k = self.dt_projs_weight[k]
            # ③ 取第k个方向的SSM参数（As/Ds/bias）
            As_k = As_list[k]  # (C, d_state)
            Ds_k = Ds_list[k]  # (C,)
            dt_projs_bias_k = dt_projs_bias_list[k]  # (C,)

            x_dbl = torch.einsum("b d l, c d -> b c l", x_cross_k, x_proj_weight_k)
            dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)

            dts = torch.einsum("b r l, d r -> b d l", dts, dt_projs_weight_k)

            out_y = self.selective_scan(
                x_k,
                dts,
                As_k,  # 单个方向的As
                Bs,
                Cs,
                Ds_k,  # 单个方向的Ds
                z=None,
                delta_bias=dt_projs_bias_k,  # 单个方向的bias
                delta_softplus=True,
                return_last_state=False,
            )  # (B, C, L)
            direction_outs.append(out_y)

        # 堆叠四向输出 + 聚合
        out_y = torch.stack(direction_outs, dim=1)  # (B, K, C, L)
        y = out_y.mean(dim=1)  # (B, C, L)：平均四向结果
        y = y.view(B, C, D, H, W)  # 恢复3D形状
        y = y.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)
        y = self.out_norm(y)

        return y


    def forward(self, x: torch.Tensor, x_cross: torch.Tensor, **kwargs):

        x = x.permute(0, 2, 3, 4, 1).contiguous()  # x.shape → (4,64,64,64,48)
        x_cross = x_cross.permute(0, 2, 3, 4, 1).contiguous()  # x_cross.shape → (4,64,64,64,48)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x_cross = self.in_proj_cross(x_cross)

        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.act(self.conv3d(x))

        y = self.forward_core(x, x_cross.permute(0, 4, 1, 2, 3))

        y = y * torch.sigmoid(z)
        out = self.out_proj(y)
        return out


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.drop=nn.Dropout(0.1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)
        self.bat=nn.BatchNorm3d(hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x=self.drop(x)
        x = self.fc2(x)
        x=self.bat(x)
        return x


class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.BatchNorm3d(in_channles)
        self.nonliner = nn.LeakyReLU()
        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.BatchNorm3d(in_channles)
        self.nonliner2 = nn.LeakyReLU()
        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.BatchNorm3d(in_channles)
        self.nonliner3 = nn.LeakyReLU()
        self.drop1=nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)

    def forward(self, x):
        x_residual = x
        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1=self.drop1(x1)
        x1=self.nonliner(x1)
        x1 = self.proj2(x1)
        x1 = self.nonliner2(x1)
        x1=self.drop2(x1)
        x2 = self.proj3(x)
        x = x1 + x2 +x_residual
        x=self.norm3(x)
        return x

class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i])
            stage = nn.Sequential(
                *[MambaLayer(d_model=dims[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices
        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 1 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

from torch.nn import init, Parameter

class GLSurv(nn.Module):
    def __init__(
            self,
            in_chans=4,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,
            spatial_dims=3,
            norm_name="instance",
            conv_block: bool = True,
            res_block: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.ln=nn.LayerNorm(768)
        self.spatial_dims = spatial_dims
        self.vit = MambaEncoder(in_chans,
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
                                )

        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.relu = nn.LeakyReLU()
        self.cssd1d=CSS1D()
        self.y = nn.Parameter(
            torch.randn(4, 768) * (1.0 / math.sqrt(768))
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.sig = nn.Sigmoid()
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, x_in):
        outs = self.vit(x_in)
        enc_hidden = self.encoder5(outs[3])
        enc_hidden = F.adaptive_avg_pool3d(enc_hidden, output_size=(1, 1, 1))
        enc_hidden = enc_hidden.view(enc_hidden.size(0), -1)
        y=self.y
        enc_hidden=self.cssd1d(enc_hidden,y)
        enc_hidden = self.ln(enc_hidden)
        x = self.fc1(enc_hidden)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.fc3(x)
        x = self.sig(x)
        x = x * self.output_range + self.output_shift
        return x,y


class CSS1D(nn.Module):
    """1D版本交叉特征融合模块，基于SSM（状态空间模型），仅四向交互
    输入：两个1D序列张量 x, x_cross → 均为 (B, L) 形状
    输出：融合后的1D特征 → (B, L) 形状
    """

    def __init__(
            self,
            d_model=64,  # 1D序列的特征维度（输入投影后的通道数）
            d_state=16,  # SSM状态维度
            d_conv=3,  # 1D卷积核大小
            ssm_ratio=2,  # 特征扩展比例（d_inner = d_model * ssm_ratio）
            dt_rank="auto",  # dt投影的秩
            dropout=0.1,  # dropout概率
            conv_bias=True,  # 卷积层偏置
            bias=False,  # 线性层偏置
            dtype=None,  # 数据类型
            dt_min=0.001,  # dt最小值
            dt_max=0.1,  # dt最大值
            dt_init="random",  # dt初始化方式
            dt_scale=1.0,  # dt缩放系数
            dt_init_floor=1e-4,  # dt初始化下限
            shared_ssm=False,  # 是否共享SSM参数
            softmax_version=False,  # 是否使用softmax版本
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": dtype}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model  # 1D特征投影维度（原3D的通道数）
        # 自动计算d_state（若未指定）
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)  # 扩展后的特征维度
        # 自动计算dt_rank（若未指定）
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.K = 4 if not shared_ssm else 1  # 四向交互，故K=4

        self.in_proj = nn.Linear(1, self.d_inner, bias=bias, **factory_kwargs)  # 单特征→d_inner通道
        self.in_proj_cross = nn.Linear(1, self.d_inner, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,  # 深度可分离：每个通道独立卷积
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,  # 保证卷积后序列长度不变
            **factory_kwargs,
        )
        self.act = nn.SiLU()  # 激活函数

        # ========== 保留SSM参数生成层（四向独立参数） ==========
        self.x_proj = [
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0))  # (4, dt_rank+2*d_state, d_inner)
        del self.x_proj  # 删除原列表，减少内存占用

        # dt参数投影层（四向独立参数）
        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0))  # (4, d_inner, dt_rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (4, d_inner)
        del self.dt_projs  # 删除原列表

        # SSM核心状态参数（A_logs、Ds）
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True)  # (4*d_inner, d_state)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True)  # (4*d_inner,)

        # 输出归一化与投影
        if not self.softmax_version:
            self.out_norm = nn.LayerNorm(self.d_inner)  # 特征归一化
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None  # 输出dropout
        # 最终投影回1D序列：(B, d_inner, L) → (B, L)
        self.out_proj = nn.Linear(self.d_inner, 1, bias=bias, **factory_kwargs)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        """初始化dt投影层，确保F.softplus(dt_bias)在[dt_min, dt_max]范围内"""
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        # 初始化权重：保持方差一致
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"不支持的dt初始化方式：{dt_init}")

        # 初始化偏置：用softplus逆运算确保初始dt在合理范围
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # softplus逆运算（参考PyTorch#72759）
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True  # 标记为无需重新初始化
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        """初始化SSM的A矩阵（取对数，保证数值稳定）"""
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",  # 每个通道对应一组1~d_state的序列
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # 取对数，后续用-exp(A_log)得到负的A矩阵（保证SSM稳定性）
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)  # 扩展为K=4组
            if merge:
                A_log = A_log.flatten(0, 1)  # 展平为(4*d_inner, d_state)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True  # 标记为无需权重衰减
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        """初始化SSM的跳跃连接参数D"""
        D = torch.ones(d_inner, device=device)  # 初始为1，保证跳跃连接有效
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)  # 扩展为K=4组
            if merge:
                D = D.flatten(0, 1)  # 展平为(4*d_inner,)
        D = nn.Parameter(D)
        D._no_weight_decay = True  # 标记为无需权重衰减
        return D

    def generate_1d_directions(self, feat):
        """生成1D序列的四向交互序列（适配1D数据）
        输入：feat → (B, d_inner, L) （1D带通道的序列）
        输出：四向序列 → (B, 4, d_inner, L)
        四向定义：
        1. 正向：原始序列顺序
        2. 反向：序列首尾翻转
        3. 奇偶交叉：取奇数位+偶数位逆序
        4. 逆奇偶交叉：取偶数位+奇数位逆序
        """
        B, C, L = feat.shape

        # 1. 正向序列
        dir_forward = feat  # (B, C, L)

        # 2. 反向序列
        dir_backward = torch.flip(feat, dims=[-1])  # (B, C, L)

        # 3. 奇偶交叉序列
        indices_odd = torch.arange(0, L, 2, device=feat.device)  # 奇数位（0-based）
        indices_even = torch.arange(1, L, 2, device=feat.device)  # 偶数位
        indices_even_rev = indices_even.flip(dims=[0])  # 偶数位逆序
        cross_indices_1 = torch.cat([indices_odd, indices_even_rev], dim=0)[:L]  # 保证长度为L
        dir_cross1 = feat[:, :, cross_indices_1]  # (B, C, L)

        # 4. 逆奇偶交叉序列
        indices_odd_rev = indices_odd.flip(dims=[0])  # 奇数位逆序
        cross_indices_2 = torch.cat([indices_even, indices_odd_rev], dim=0)[:L]  # 保证长度为L
        dir_cross2 = feat[:, :, cross_indices_2]  # (B, C, L)

        # 堆叠四向序列：(B, 4, C, L)
        return torch.stack([dir_forward, dir_backward, dir_cross1, dir_cross2], dim=1)

    def forward_core(self, x: torch.Tensor, x_cross: torch.Tensor):
        """1D特征融合核心逻辑：四向交互+SSM长距离建模（已按方向分配权重）
        输入：
            x → (B, d_inner, L) （主输入1D特征）
            x_cross → (B, d_inner, L) （交叉输入1D特征）
        输出：融合后的特征 → (B, d_inner, L)
        """
        self.selective_scan = selective_scan_fn
        B, C, L = x.shape
        K = self.K  # 四向交互

        # 生成1D四向序列
        xs = self.generate_1d_directions(x)  # (B, 4, C, L)
        xs_cross = self.generate_1d_directions(x_cross)  # (B, 4, C, L)

        # 1. 拆分As：(4*C, d_state) → 4个 (C, d_state)
        As = -torch.exp(self.A_logs.float())  # (4*C, d_state)
        As_list = torch.split(As, C, dim=0)  # [As_0, As_1, As_2, As_3]

        # 2. 拆分Ds：(4*C,) → 4个 (C,)
        Ds = self.Ds.float()  # (4*C,)
        Ds_list = torch.split(Ds, C, dim=0)  # [Ds_0, Ds_1, Ds_2, Ds_3]

        # 3. 拆分dt_projs_bias：(4, C) → 展平为(4*C,) → 拆分为4个 (C,)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (4*C,)
        dt_projs_bias_list = torch.split(dt_projs_bias, C, dim=0)  # [bias_0, bias_1, bias_2, bias_3]

        direction_outs = []
        for k in range(K):
            # 取第k个方向的特征：(B, C, L)
            x_k = xs[:, k, :, :].float()  # (B, C, L)
            x_cross_k = xs_cross[:, k, :, :].float()  # (B, C, L)
            # 1. 取第k个方向的x_proj_weight → (c, d_inner)（2维，匹配einsum方程）
            x_proj_weight_k = self.x_proj_weight[k]  # (dt_rank+2*d_state, d_inner)
            # 2. 取第k个方向的dt_projs_weight → (d_inner, dt_rank)（2维，匹配dt投影）
            dt_projs_weight_k = self.dt_projs_weight[k]  # (d_inner, dt_rank)

            # 交叉输入驱动SSM参数生成（dts、Bs、Cs）
            # einsum：(B, C, L) × (c, C) → (B, c, L)（2维×2维，下标数匹配）
            x_dbl = torch.einsum("b d l, c d -> b c l", x_cross_k, x_proj_weight_k)
            # split：按dt_rank/d_state/d_state拆分，sum=dt_rank+2*d_state，和x_dbl.dim=1匹配
            dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
            # dt参数投影：(B, dt_rank, L) × (C, dt_rank) → (B, C, L)
            dts = torch.einsum("b r l, d r -> b d l", dts, dt_projs_weight_k)
            out_y = self.selective_scan(
                x_k,
                dts,
                As_list[k],  # 第k个方向的As
                Bs,
                Cs,
                Ds_list[k],  # 第k个方向的Ds
                z=None,
                delta_bias=dt_projs_bias_list[k],  # 第k个方向的bias
                delta_softplus=True,
                return_last_state=False,
            )  # (B, C, L)
            direction_outs.append(out_y)

        # 堆叠四向输出：(B, 4, C, L)
        out_y = torch.stack(direction_outs, dim=1)  # (B, K, C, L)
        # 四向特征聚合（平均融合，保留所有方向信息）
        y = out_y.mean(dim=1)  # (B, C, L)：平均四向结果
        return y


    def forward(self, x: torch.Tensor, x_cross: torch.Tensor) -> torch.Tensor:
        """
        前向传播：处理1D序列输入
        输入：
            x: (B, L) → 主输入1D序列
            x_cross: (B, L) → 交叉输入1D序列
        输出：
            out: (B, L) → 融合后的1D序列
        """
        # 维度校验
        assert x.dim() == 2 and x_cross.dim() == 2, "输入必须是2维张量 (B, L)"
        assert x.shape == x_cross.shape, "两个输入的形状必须一致"
        B, L = x.shape

        # Step1：将1D序列 (B, L) → 扩展为 (B, L, 1) → 逐位置投影到d_inner通道 → (B, L, d_inner) → 转置为 (B, d_inner, L)
        x_reshaped = x.unsqueeze(-1)  # (B, L, 1)
        x_proj = self.in_proj(x_reshaped).permute(0, 2, 1).contiguous()  # (B, d_inner, L)
        x_proj = self.act(self.conv1d(x_proj))  # 1D卷积提取局部特征

        x_cross_reshaped = x_cross.unsqueeze(-1)  # (B, L, 1)
        x_cross_proj = self.in_proj_cross(x_cross_reshaped).permute(0, 2, 1).contiguous()  # (B, d_inner, L)
        x_cross_proj = self.act(self.conv1d(x_cross_proj))  # 1D卷积提取局部特征
        y = self.forward_core(x_proj, x_cross_proj)  # (B, d_inner, L)

        y = y.permute(0, 2, 1).contiguous()  # (B, L, d_inner)
        y = self.out_norm(y)  # 归一化
        if self.dropout is not None:
            y = self.dropout(y)
        out = self.out_proj(y).squeeze(-1)  # (B, L, 1) → (B, L)

        return out
