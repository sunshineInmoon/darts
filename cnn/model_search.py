import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module): # 计算一条path上所有op的加权和

  def __init__(self, C, stride): #注意C是Channels，stride是卷积中的参数
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES: #genotypes.py，候选操作名称集合
      op = OPS[primitive](C, stride, False) #OPS在operations.py中，具体一个操作
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops)) #所有op的加权和


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction #指明当前cell是否是reduction cell，注意reduction，reduction_prev不可能同时为真

    if reduction_prev: #指明它的前一个Cell是不是reduction cell，即输出H，W减少一倍，Channels增加一倍
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False) #对前前一个Cell做降维处理，？？为什么要做这样一个处理呢？？？
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False) #对前前一个Cell的输出做简单的卷积操作，注意kernel=1，stride=1，也就是一个Channels的转换
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False) #对前一个Cell的输出只做1X1卷积操作
    self._steps = steps # 4
    self._multiplier = multiplier # 4

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps): #self._steps=4，一个cell里共有6个Node，self._steps+2个input；这里的i循环的是负责计算的Node
      for j in range(2+i): #每个Node的输入都是之前所有的Node的输出，除了Cell的两个输入，这里的2就是那两个输入；j负责循环的是输入到该Node的path
        stride = 2 if reduction and j < 2 else 1 # reduction=true，标明该cell是reduction cell，feature map的h，w需要减半；j<2,目的让该cell的两个输入降维，因为之前的Node已经在前面的循环中降维了
        op = MixedOp(C, stride)
        self._ops.append(op)# self._ops 保存的是所有path

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0) # 对前前cell的输出预处理
    s1 = self.preprocess1(s1) # 对前cell的输出预处理

    states = [s0, s1] #states 保存的是当前Node的输入，即之前所有Node的输出
    offset = 0 # 由于self._ops是把所有的path保存成一个list，因此需要一个offset去定位，属于该Node的所有输入path的首个
    for i in range(self._steps): #self._steps = 4
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states)) #states随着循环会不断增加，应为有新Node被计算了
      offset += len(states)
      states.append(s) 

    return torch.cat(states[-self._multiplier:], dim=1) #states保存着所有Node的输出，从参数上来看只去最后4个Node的输出，并在Channels维度链接,因此主要注意这里的输出维度self._multiplier*C


class Network(nn.Module):
  # model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  # args.init_channels = 16
  # CIFAR_CLASSES = 10
  # args.layers = 8
  # criterion 交叉熵
  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C # 16
    self._num_classes = num_classes # 10
    self._layers = layers # 8 Net中有个8个Cell
    self._criterion = criterion# 交叉熵
    self._steps = steps # 4 一个cell中有4个计算Node
    self._multiplier = multiplier # 4

    # 网络的head是固定的不用搜索，输入3Channels的彩色图，输出3*16=48Channels的feature map
    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    # 每一个Cell都有两个输入，来此前前Cell和前Cell
    # 对第一个cell来说，它的前前Cell和前Cell是同一个，都是网络的head
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C # C_curr=48，C=16
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers): # layers=8
      if i in [layers//3, 2*layers//3]: #[layers//3, 2*layers//3]=[2,5]，即在第3，6cell中需要H，W减少，Channels增加
        C_curr *= 2 # Channles 增加一倍
        reduction = True # 空间分辨率减少一倍
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev) #steps=4
      reduction_prev = reduction
      self.cells += [cell] #加入cells列表
      C_prev_prev, C_prev = C_prev, multiplier*C_curr #由于当前Cell的输出是multiplier条path在Channels维度上cat，因此输出Channels=multiplier*C_curr   

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes) #classifier居然是个fc层，应该是fc2

    self._initialize_alphas() #初始化

  def new(self): #更新结构参数，arch_parameters, 应该是交替训练时使用
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new #返回一个net weight随机，architecture parameters被更新的新网络

  def forward(self, input):
    s0 = s1 = self.stem(input) #输入图片，首先送入网络head
    for i, cell in enumerate(self.cells): #遍历cell
      if cell.reduction: #根据cell类型，选择weights类型
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights) #精妙
    out = self.global_pooling(s1) #要对最后的输出做一个全局pooling
    logits = self.classifier(out.view(out.size(0),-1)) #fc2层的输出
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) # 计算最终的交叉熵loss

  def _initialize_alphas(self): #初始化一个Cell的architecture parameters
    k = sum(1 for i in range(self._steps) for n in range(2+i)) #所有cell中所有path条数, self._steps=4, k=14
    num_ops = len(PRIMITIVES) # 候选操作数量，一条path上；有num_ops个操作，因此architecture parameters是个k*num_ops的矩阵

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True) # normal cell使用, 矩阵中每一行代表一条path，每一列代表一个path上的一个op的权重
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True) # reduce cell使用
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self): #获取结构参数
    return self._arch_parameters

  def genotype(self): #按照概率挑选最佳op

    def _parse(weights): #操作范围还是在一个cell中
      gene = []
      n = 2 #两个输入Node
      start = 0
      for i in range(self._steps): #self._steps=4,从这个参数可以看出这是对一个cell进行的操作
        end = start + n
        W = weights[start:end].copy() #一个cell中所有path组成一个矩阵，取出对应path的权重
        # 每个Node的输入是前面所有Node输出（包括两个输入Node），下面的作用是从所有输入path中找到两条包含最大概率op的path，具体分析如下
        # 可以看到排序的主体是range(i + 2)
        # 注意lambda表达式，出入的参数x in range(i + 2)
        # W[x][k] for k in range(len(W[x])) 遍历矩阵W第x行，实际上是在遍历第x条path上的op权重，找出权重最大的op，max()前之所以加“-”
        # ；因为sorted函数默认升序排序，这样可以使拥有最大权重op的path排在前面
        # 注意最后的[:2]选择了两个最大的path
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2] 
        for j in edges: #遍历我们选择的两个拥有最大权重op的path
          k_best = None
          for k in range(len(W[j])): #遍历每条path上的op的权重
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k #记录最佳op
          gene.append((PRIMITIVES[k_best], j)) #第j条path，最佳op为PRIMITIVES[k_best]
        start = end # 进行下一个Node，在W中所有Node的输入path按顺序排好了
        n += 1 # 对于下一个Node，输入Node增加一个
      return gene #注意返回的格式，可参考genotypes.py中的格式

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2) #确定哪个几个Node的输出相连，作为该cell的输出
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

