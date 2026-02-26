import torch
import torch.nn as nn
from einops import repeat
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

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

    # ========== 核心修改3：3D方向生成 → 1D四向序列生成 ==========
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

class GeneEncoder(nn.Module):
    """
    基因特征编码器：将3072维基因特征逐步编码为2维特征
    - 输入: 基因特征张量 (batch_size, 3072)
    - 输出: 编码后的2维特征 (batch_size, 2)
    - 结构: 3072 → 1024 → 768 → 64 → 2（全连接层逐级降维）
    """

    def __init__(self, dropout_rate: float = 0.1):
        super(GeneEncoder, self).__init__()

        self.fc1 = nn.Linear(in_features=3072, out_features=512)
        self.drop = nn.Dropout(p=dropout_rate)
        self.fc2=nn.Linear(in_features=512, out_features=768)
        self.fc3 = nn.Linear(in_features=768, out_features=64)
        self.ln=nn.LayerNorm(768)
        self.fc4 = nn.Linear(in_features=64, out_features=2)
        self.relu=nn.LeakyReLU()
        self.cssd=CSS1D()
        self.l=nn.LayerNorm(512)

    def forward(self, x,y):
        self.drop(x)
        x = self.fc1(x)
        x=self.l(x)
        x=self.drop(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.cssd(y,x)
        x=self.ln(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc4(x)
        return x




