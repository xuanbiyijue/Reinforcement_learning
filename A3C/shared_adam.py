import torch

#params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
#lr (float, 可选) – 学习率（默认：1e-3）
#betas (Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999）
#eps (float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
#weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）


class SharedAdam(torch.optim.Adam):       #共享优化器
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,weight_decay=0):  #
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()