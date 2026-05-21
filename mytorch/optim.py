import math
from collections import defaultdict
from collections import Counter
from .tensor import Tensor
class Optimizer:
    def __init__(self, params,defaults):
        self.defaults=defaults  # 将传入的defaults保存为类属性,用于后续为参数组设置默认值
        self.param_groups=[]  # 初始化为一个空列表，用于后续存储所有的参数组（每个参数组是一个字典），实现参数分组功能，为什么？
        self.state=defaultdict(dict)  # 初始化state为defaultdict(dict)（defaultdict能确保当访问不存在的参数时，自动创建空字典）,用于存储不同的优化器的每个参数在优化过程中的状态
        param_groups=list(params)  # 初始化一个列表，用于处理传入的params，后续会将其中的数据通过add函数传入self.param_groups
        if not isinstance(param_groups[0], dict):
            param_groups=[{'params': param_groups}]
        for group in param_groups:
            self.add_param_group(group)
    def add_param_group(self,param_group):
        params=param_group['params']  # 提取参数列表
        '''
        确保param_group['params']是列表，用于后续step（）方法遍历参数列表
        '''
        if isinstance(params, Tensor):
            param_group['params']=[params]  # 如果param_group['params']只有一个参数，将其转换为列表
        else:
            param_group['params']=list(params)  # 多个参数
        self.param_groups.append(param_group)  # 将处理过后的参数组添加到优化器的self.param_roups列表中
        for name, default in self.defaults.items():
            param_group.setdefault(name, default)  # 合并传入参数和默认参数，如果参数组中已有name键则保持原值，如果没有则自动添加新值
    def zero_grad(self):
        for group in self.param_groups:
            for param in group['params']:
                param.zero_grad()
    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self,params,lr=0.001,weight_decay=0):
        defaults=dict(lr=lr,weight_decay=weight_decay)
        super().__init__(params,defaults)
    def step(self):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                d_param=param.grad
                if group['weight_decay'] != 0:
                    d_param.add_(param,alpha=group['weight_decay'])
                param.add_(d_param,alpha=-group['lr'])

class Momentum(Optimizer):
    def __init__(self,params,lr=0.001,beta=0.9,weight_decay=0):
        defaults = dict(lr=lr, beta=beta,weight_decay=weight_decay)
        super().__init__(params,defaults)
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]  # 创建指向self.state[param]的引用，state指向self.state[param]
                state['v']=Tensor.full_like(param,0.0)
    def step(self):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                d_param=param.grad
                if group['weight_decay'] != 0:
                    d_param.add_(param,alpha=group['weight_decay'])
                state = self.state[param]
                v = state['v']
                v.mul_(group['beta'])
                v.add_(d_param,alpha=1)
                param.add_(v,alpha=-group['lr'])

class Adagrad(Optimizer):
    def __init__(self,params,lr=0.01,initial_sum=0,eps=1e-10,weight_decay=0):
        defaults = dict(lr=lr,weight_decay=weight_decay,eps=eps,initial_sum=initial_sum)
        super().__init__(params,defaults)
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['sum']=Tensor.full_like(param,group['initial_sum'])
    def step(self):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                d_param=param.grad
                if group['weight_decay'] != 0:
                    d_param.add_(param,alpha=group['weight_decay'])
                state = self.state[param]
                sum=state['sum']
                sum.addcmul_(d_param,d_param,value=1)
                std=sum.sqrt().add_(group['eps'])
                param.addcdiv_(d_param,std,value=-group['lr'])

class Rmsprop(Optimizer):
    def __init__(self,params,lr=0.01,initial_sum=0,eps=1e-10,weight_decay=0,beta=0.9):
        defaults = dict(lr=lr, initial_sum=initial_sum,eps=eps, weight_decay=weight_decay, beta=beta)
        super().__init__(params,defaults)#调用父类方法初始化
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['sum'] = Tensor.full_like(param, group['initial_sum'])
    def step(self):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                d_param = param.grad
                if group['weight_decay'] != 0:
                    d_param.add_(param, alpha=group['weight_decay'])
                state = self.state[param]
                sum = state['sum']
                sum.mul_(group['beta'])
                sum.addcmul_(d_param, d_param, value=1-group['beta'])
                std = sum.sqrt().add_(group['eps'])
                param.addcdiv_(d_param, std, value=-group['lr'])
class Adam(Optimizer):
    def __init__(self,params,lr=0.01,eps=1e-10,weight_decay=0,betas=(0.9, 0.999)):
        defaults=dict(lr=lr ,eps=eps, weight_decay=weight_decay, beta1=betas[0],beta2=betas[1])
        super().__init__(params,defaults)
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['step'] = 0
                state['sum'] = Tensor.full_like(param,0.0)
                state['v'] = Tensor.full_like(param, 0.0)
    def step(self):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                d_param=param.grad
                if group['weight_decay'] != 0:
                    d_param.add_(param, alpha=group['weight_decay'])
                self.state[param]['step'] += 1
                v=self.state[param]['v']
                v.mul_(group['beta1'])
                v.add_(d_param, alpha=1-group['beta1'])
                sum=self.state[param]['sum']
                sum.mul_(group['beta2'])
                sum.addcmul_(d_param, d_param, value=1 - group['beta2'])
                bias_correction1 = 1 - (group['beta1'] ** self.state[param]['step'])
                bias_correction2 = 1 - (group['beta2'] ** self.state[param]['step'])
                x= group['lr'] * (math.sqrt(bias_correction2) / bias_correction1)
                std = sum.sqrt().add_(group['eps'])
                param.addcdiv_(v, std, value=-x)



class LRScheduler:
    def __init__(self,optimizer:Optimizer,last_epoch:int =-1,verbose:bool=False):
        self.optimizer=optimizer
        '''
        如果last_epoch==1,则表示
        '''
        if last_epoch==-1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr',group['lr'])#检查group中是否存在key“initial_lr”,如果不存在则添加并设值为group['lr']
            else:
                for i, group in enumerate(optimizer.param_groups):
                    if 'initial_lr' not in group:
                        raise KeyError(f"param 'initial_lr' not found in optimizer"
                                       "in param_groups[{i}] when resuming an optimizer")
            self.base_lrs=[group['initial_lr'] for group in optimizer.param_groups]
            self.last_epoch = last_epoch
            self.verbose = verbose
            self._initial_step()
    def get_lr(self):
        return NotImplementedError

    def get_last_lr(self):
        return self._last_lr
    def print_lr(self, is_verbose, group, lr, epoch=None):
        if is_verbose:
            if epoch is None:
                print(f"Adjusting learning rate of group {group} to {lr:.4e}.")
            else:
                epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
                print(f'Epoch {epoch_str}: adjusting learning rate of group {group} to {lr:.4e}.')
    def _initial_step(self):
        """初始化step count并调用一次step"""
        self.optimizer._step_count = 0
        self._step_count = 0
        self.step()

    def step(self, epoch=None):
        self._step_count += 1
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
            param_group, lr = data
            param_group["lr"] = lr  # 用新的学习率覆盖当前学习率
            self.print_lr(self.verbose, i, lr, epoch)
        # 保存最近一次学习率
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


#%%
'''
指数衰减学习率
'''
class ExponentialLR(LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1, verbose=False):
        """
        每个epoch通过gamma衰减每个parameter group的学习率，当last_epoch=-1，学习率设为初始值
        :param optimizer: 优化器
        :param gamma: 学习率衰减的乘法因子
        :param last_epoch: 最后一次epoch的索引
        :param verbose: 是否为每次更新打印信息
        """
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            # 第一次迭代就是初始学习率
            return [group["lr"] for group in self.optimizer.param_groups]
        # 然后是当前学习率乘以gamma
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]
#%%
'''
阶梯式学习率衰减
'''
class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False):
        """
        每step_size个epoch通过gamma衰减每个parameter group的学习率，当last_epoch=-1，学习率设为初始值

        :param optimizer:
        :param step_size:
        :param gamma:
        :param last_epoch:
        :param verbose:
        """
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            # 第一次迭代或在第一个step_size间隔内
            return [group["lr"] for group in self.optimizer.param_groups]
        # 然后是当前学习率乘以gamma
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]

#%%
'''
多阶段阶梯式学习率衰减
'''
class MultiStepLR(LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False):
        """
        一旦epoch次数达到milestones中的次数，则通过gamma衰减每个parameter group的学习率，当last_epoch=-1，学习率设为初始值

        :param optimizer:
        :param milestones: epoch索引列表，注意必须是递增的
        :param gamma:
        :param last_epoch:
        :param verbose:
        """
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch not in self.milestones:
            # 如果不在milestones内，则返回当前的学习率
            return [group["lr"] for group in self.optimizer.param_groups]
        # 然后是当前学习率乘以gamma的milestones[last_epoch]次
        return [group["lr"] * self.gamma ** self.milestones[self.last_epoch] for group in self.optimizer.param_groups]

#%%
'''
自定义函数学习率调度器
'''
class LambdaLR(LRScheduler):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
        """
        让每个parameter group的学习率为初始学习率乘以一个给定的函数lr_lambda
        :param optimizer:
        :param lr_lambda(function or list): 一个基于epoch计算乘法因子的函数；或是一个这样的函数列表，列表中每个函数
                                            对应optimizer.param_groups的每个group
        :param last_epoch:
        :param verbose:
        """
        self.optimizer = optimizer

        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            # 如果是列表的话必须和param_groups的大小一致
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} lr_lambdas, but got {len(lr_lambda)}")
            self.lr_lambdas = list(lr_lambda)

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [base_lr * lmbda(self.last_epoch) for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]
#%%
'''
余弦退火学习率调度器
'''
class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False):
        """
        由SGDR提出，但这里仅实现余弦退火部分，并不包含热重启部分。
        Args:
            optimizer:
            T_max: 最多迭代次数
            eta_min: 最小学习率
            last_epoch:
            verbose:
        """
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            # 刚开始时，学习率最大，为默认的学习率
            return [group['lr'] for group in self.optimizer.param_groups]

        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(self.last_epoch * math.pi / self.T_max)) / 2 for
                base_lr in self.base_lrs]

#%%
class CosineAnnealingWarmRestarts(LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False):
        """
        使用余弦退火衰减调整每个参数组的学习率，并在T_i次epoch后进行热重启，重启为初始学习率。
        T_i是两次热重启之间的间隔epoch次数。
        Args:
            optimizer:
            T_0: 第一次重启的epoch次数
            T_mult: 重启周期增大因子， ≥ 1
            eta_min: 最小学习率
            last_epoch:
            verbose:
        """
        self.T_0 = T_0
        self.T_i = T_0  # 初始T_i 为 T_0 ，后面可能会增大
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch  # 当前间隔内的epoch次数
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(self.T_cur * math.pi / self.T_i)) / 2 for
                base_lr in self.base_lrs]

    def step(self, epoch=None):
        """这里需要重写step，在里面更新T_i和T_cur"""
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur > self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult  # 重启次数乘以增大因子
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            if epoch >= self.T_0:
                # 如果增大因子为1，即不增大
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    # 计算当前是第几次周期内，T_i为当期周期的大小
                    # 假设T_0=8;T_mul=2;
                    # 那么0-7属于第一次周期，该周期大小为8，epoch=0属于第一次周期的开始(更新T_cur=0)；
                    # 那么8-23属于第二次周期，该周期大小为16，epoch=24属于第二次周期的开始(更新T_cur=0)；
                    # T_cur是当期周期内的epoch数
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    # 更新当前周期内的epoch数
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    # 计算周期大小
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                # 如果还在第一个周期内
                self.T_i = self.T_0
                self.T_cur = epoch

        self.last_epoch = math.floor(epoch)

        for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
            param_group, lr = data
            param_group["lr"] = lr
            self.print_lr(self.verbose,i,lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
#%%
class NoamLR(LRScheduler):
    def __init__(self, optimizer, model_size, factor=1., warmup_steps=4000, last_epoch=-1, verbose=False):
        """
        参考 http://nlp.seas.harvard.edu/annotated-transformer 实现的Transformer提出的学习率衰减方法
        在第一个warmup_steps内线性地增大学习率，然后按步长的平方倒数成比例地减小
        :param optimizer: 优化器
        :param model_size: 模型嵌入层大小
        :param factor: 乘法因子
        :param warmup_steps: 加热步
        :param last_epoch:
        :param verbose:
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.model_size = model_size
        self.factor = factor

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        # 避免0的负幂次
        if self.last_epoch == 0:
            self.last_epoch = 1

        step = self.last_epoch
        lr = self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))
        return [lr] * len(self.optimizer.param_groups)
#%%
class ReduceLROnPlateau:
    def __init__(self, optimizer, mode="min", factor=0.1, patience=10, threshold=1e-4, threshold_mode="rel", cooldown=0,
                min_lr=0, eps=1e-8, verbose=False):
        """
        当某个指标在一定返回的epoch内(patience)停止提升时才进行学习率衰减，避免偶发的指标为提升导致的学习率衰减
        Args:
            optimizer:
            mode: min|max，指标是越小越好，还是越大越好
            factor: 衰减的乘法因子 < 1
            patience: 能容忍多少次指标不提升
            threshold: 至少提升了threshold才认为是真的提升，默认为1e-4
            threshold_mode: rel|abs。在rel模式下，max方式下dynamic_threshold = best * ( 1 + threshold )，
                                                min方式下，dynamic_threshold = best * ( 1 - threshold )；
                                     在abs模式下，max方式下dynamic_threshold = best + threshold，
                                                min方式下dynamic_threshold = best - threshold。

            cooldown: 进行一次学习率衰减后，多少个epoch内不继续衰减
            min_lr: 学习率的最小下限
            eps: 学习率的最小衰减值，如果衰减前后学习率的差值小于eps，那么就不进行更新
            verbose:

        Returns:

        """

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        self.optimizer = optimizer
        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(f"expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}")
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # 选定mode的更差的值
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch

        # 如果当期指标比最佳的好
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        # 在cooldown_counter > 0时不会进行衰减
        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # 在cooldown期间内num_bad_epoch一直为0

        if self.num_bad_epochs > self.patience:
            # 如果差的epoch次数大于容忍的次数，则进行学习率衰减
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown  # 进入cooldown期间
            self.num_bad_epochs = 0  # 重置为0

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            # 设定新的学习率，但不能小于预设的最小学习率
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            # 如果new_lr确实减少了
            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr
                if self.verbose:
                    epoch_str = (f"{epoch:.2f}" if isinstance(epoch, float) else f"{epoch:.5d}")
                    print(f"Epoch {epoch_str}: reducing learning rate  of group {i} to {new_lr:.4e}.")

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        """ 判断a是否比best要好"""
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1 - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == "max" and epsilon_mode == "abs":
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = -float('inf')

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode


