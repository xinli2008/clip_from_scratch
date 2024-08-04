import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from clip import Clip
from dataset import MNIST
import os
from torch.utils.tensorboard import SummaryWriter

# config
device = "cuda:0" if torch.cuda.is_available() else "cpu"
max_iter = 10000
learning_rate = 1e-3
batch_size = 64
target_count = 10
save_step = 1000
log_dir = "./logs"
model_dir = "./models"

# create folder
for item in (log_dir, model_dir):
    if not os.path.exists(item):
        os.makedirs(item)

# model、dataloader and optimizer
my_dataset = MNIST()
model = Clip()
model = model.to(device)
dataloader = DataLoader(dataset = my_dataset, batch_size = batch_size, shuffle = True, num_workers = 8, pin_memory = True, drop_last = True)
optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()))

r"""
    Question1: what is the difference between model.train and model.eval、with torch.no_grad()
        Refs: https://www.acwing.com/blog/content/29838/
        model.train()和model.eval()的不同, 在于两种模式对batch-norm和dropout的不同。
        
        model.train(): 使用batch normalization和dropout
            在训练过程中, model.train()用来启动batch normalization和dropout。
            对于batch normalization, model.train()会计算每个mini-batch数据的均值和方差, 并用于归一化数据, 以确保训练过程中每一批数据的均值和方差都可以被batch normalization用到。
            对于dropout层, model.train()会随机舍弃一部分神经元, 以避免过拟合, 并在反向传播中更新参数。
        
        model.eval(): 不使用batch normalization和dropout
            在测试和推理过程中, 需要关闭batch normalization和dropout。
            对于Batch Normalization 层,model.eval() 使用训练过程中累积的整个数据集的均值和方差来归一化数据,而不是使用每个批次数据的均值和方差。
            对于Dropout层,model.eval() 不会随机舍弃神经元,而是利用所有的神经元进行前向传播,以获得稳定的输出结果。

    Question2: what is the differene between model.eval and with torch.no_grad()？
        torch.no_grad()：用于上下文中禁用梯度计算，仅影响梯度计算，不会改变模型的状态或层的行为，主要用于推断过程中减少内存消耗或提高速度。
"""
# train mode
model.train()

# tensorboard writer
writer = SummaryWriter(log_dir)

# train
for i in range(max_iter):
    while True:
        ### dataloader是一个迭代器, 它实现了__iter__和__next__方法, 因此可以使用next(iter(iterator))获取下一次需要获取的数据
        ### 通过iter(dataloader)可以得到一个迭代器对象, 然后使用next函数来获取下一个batch的数据
        image, label = next(iter(dataloader))
        if torch.unique(label).shape[0] < target_count:
            continue
        target = set()
        indexes = []
        for j in range(batch_size):
            if label[j].item() in target:
                continue
            target.add(label[j].item())
            indexes.append(j)
            
            # 取到符合条件的target_num个数据
            if len(target) == target_count:
                break
        batch_image = image[indexes]
        batch_text  = label[indexes]
        break
    
    logits=model(batch_image.to(device),batch_text.to(device))
    
    targets=torch.arange(0, target_count).to(device)

    # 交叉熵损失来计算loss
    loss_i=F.cross_entropy(logits,targets)
    loss_t=F.cross_entropy(logits.permute(1,0),targets)
    loss=(loss_i+loss_t)/2
    print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # log loss to TensorBoard
    writer.add_scalar('Loss/train', loss.item(), i)

    # save model every 1000 steps
    if (i + 1) % save_step == 0:
        model_path = os.path.join(model_dir, f"model_step_{i + 1}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at step {i + 1}")

# close the TensorBoard writer
writer.close()

