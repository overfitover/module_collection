import torch
from torch.autograd import Variable
import torch.nn as nn

# nn.L1Loss: loss(input, target)=|input-target|  1-1
if False:
    loss_fn = torch.nn.L1Loss(reduce=True, size_average=False)
    input = torch.autograd.Variable(torch.randn(3, 4))
    target = torch.autograd.Variable(torch.randn(3, 4))
    loss = loss_fn(input, target)
    print(input)
    print(target)
    print(loss)
    print(input.size(), target.size(), loss.size())


# nn.SmoothL1Loss 　在(-1, 1)上是平方loss, 其他情况是L1 loss  1-1
if True:
    loss_fn = torch.nn.SmoothL1Loss(reduce=True, size_average=True)
    input = torch.autograd.Variable(torch.randn(3, 4))
    target = torch.autograd.Variable(torch.randn(3, 4))
    loss = loss_fn(input, target)
    print(input)
    print(target)
    print(loss)
    print(input.size(), target.size(), loss.size())

# nn.MSELoss  均方损失函数  1-1
if False:
    loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
    input = torch.autograd.Variable(torch.randn(3, 4))
    target = torch.autograd.Variable(torch.randn(3, 4))
    loss = loss_fn(input, target)
    print(input)
    print(target)
    print(loss)
    print(input.size(), target.size(), loss.size())

# nn.BCELoss  1-1  适用于二分类问题
if False:
    import torch.nn.functional as F

    loss_fn = torch.nn.BCELoss(reduce=False, size_average=False)
    input = torch.autograd.Variable(torch.randn(3, 4))
    target = torch.autograd.Variable(torch.FloatTensor(3, 4).random_(2))
    loss = loss_fn(F.sigmoid(input), target)
    print(input, input.shape)
    print(F.sigmoid(input))
    print(target, target.shape)
    print(loss, loss.shape)


# nn.CrossEntropyLoss   （3, 5）-（3）　适用于多分类问题
if False:
    weight = torch.Tensor([1, 2, 1, 1, 10])
    loss_fn = torch.nn.CrossEntropyLoss(reduce=False, size_average=False, weight=None)
    input = Variable(torch.randn(3, 5))  # (batch_size, C)
    target = Variable(torch.LongTensor(3).random_(5))   # target 减少channel维度
    loss = loss_fn(input, target)
    print(input)
    print(target)
    print(loss)

# nn.NLLLoss 负的　log likehood loss损失.用于训练一个ｎ类分类器.
if False:
    m = nn.LogSoftmax()
    loss = nn.NLLLoss()
    # input is of size nBatch x nClasses = 3 x 5
    input = Variable(torch.randn(3, 5), requires_grad=True)
    # each element in target has to have 0 <= value < nclasses
    target = Variable(torch.LongTensor([1, 0, 4]))
    output = loss(m(input), target)
    print(output)

# nn.LLLoss2d  对于图片的　negtive log likehood loss.计算每个像素的NLL loss
if False:
    m = nn.Conv2d(16, 32, (3, 3)).float()
    loss = nn.NLLLoss2d()
    # input is of size nBatch x nClasses x height x width
    input = Variable(torch.randn(3, 16, 10, 10))
    # each element in target has to have 0 <= value < nclasses
    target = Variable(torch.LongTensor(3, 8, 8).random_(0, 4))
    output = loss(m(input), target)
    print(m(input))
    print(output)

# nn.MultiLabelMarginLoss  input : x --> (N, c), y --> (N, c)其中y是 LongTensor, 且其元素为类别的index
if False:
    x=Variable(torch.randn(3, 4))
    y=Variable(torch.LongTensor(3, 4).random_(4))
    loss=torch.nn.MultiLabelMarginLoss()
    output = loss(x, y)
    print(output)

# nn.MultiLabelSoftMarginLoss  与MultiLableMarginLoss相同，区别在于y的类型是FloatTensor
if False:
    x = Variable(torch.randn(3, 10))
    y = Variable(torch.FloatTensor(3, 10).random_(10))
    loss = torch.nn.MultiLabelSoftMarginLoss()
    output = loss(x, y)
    print(output)

# nn.MultiMarginLoss 适用于多分类模型
if False:
    x = Variable(torch.randn(3, 10))
    y = Variable(torch.LongTensor(3).random_(10))
    loss = torch.nn.MultiMarginLoss()
    output=loss(x, y)
    print(output)




