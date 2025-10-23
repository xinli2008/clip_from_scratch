import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from clip import Clip
from dataset import MNIST
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging

def parse_args():
    parser = argparse.ArgumentParser(description='Train CLIP model on MNIST')
    
    # Model parameters
    parser.add_argument('--model_dir', type=str, default='./models', 
                       help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for tensorboard logs')
    
    # Training parameters
    parser.add_argument('--max_iter', type=int, default=10000,
                       help='Maximum training iterations')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--target_count', type=int, default=10,
                       help='Number of unique classes per batch')
    
    # Logging and saving
    parser.add_argument('--save_step', type=int, default=1000,
                       help='Save model every N steps')
    parser.add_argument('--log_step', type=int, default=100,
                       help='Log training info every N steps')
    
    # System parameters
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of workers for data loading')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_directories(args):
    """Create necessary directories"""
    for dir_path in [args.log_dir, args.model_dir]:
        os.makedirs(dir_path, exist_ok=True)

def setup_model_and_data(args):
    """Setup model, dataset, and dataloader"""
    # Determine device
    if args.device == 'auto':
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Dataset and dataloader
    dataset = MNIST()
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=True
    )
    
    # Model
    model = Clip()
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.learning_rate
    )
    
    return model, dataloader, optimizer, device

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model and optimizer state from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_iter = checkpoint.get('iteration', 0)
    return start_iter

def save_checkpoint(model, optimizer, iteration, save_path):
    """Save model and optimizer state"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, save_path)

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

    Question3: Clip训练时候的loss是如何计算的？
        Clip模型的训练目标是让匹配的数据对之间的相似度更高，而不匹配的数据对之间的相似度更低。
        它通过对比学习来计算损失, 具体的做法是clip在得到了图像和文本的特征表示后, 计算它们之间的相似度矩阵, 然后使用对比损失函数来优化模型参数。
        损失函数通常采用交叉熵损失（Cross-Entropy Loss），分别计算图像到文本和文本到图像的损失，然后将两者平均作为最终的损失值。这种双向的损失计算方式有助于模型更好地学习图像和文本之间的对应。

    Question4: Clip的温度参数temperature有什么作用？
        在CLIP模型中，温度参数（temperature）用于调整图像和文本特征之间相似度的分布。具体来说，温度参数通过缩放相似度分数来控制模型对不同相似度值的敏感度。
        较低的温度值会使得相似度分布更加尖锐，增强模型对高相似度对的区分能力，从而使得模型更倾向于关注最相关的图像-文本对。相反，较高的温度值会使得相似度分布更加平缓，降低模型对高相似度对的区分能力，从而使得模型在训练过程中更加关注整体的相似度
    
"""

def train_model(args):
    """Main training function"""
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting training with args: {args}")
    
    # Create directories
    create_directories(args)
    
    # Setup model and data
    model, dataloader, optimizer, device = setup_model_and_data(args)
    
    # Resume from checkpoint if specified
    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(model, optimizer, args.resume)
        logger.info(f"Resumed training from iteration {start_iter}")
    
    # Set model to train mode
    model.train()
    
    # Setup tensorboard writer
    writer = SummaryWriter(args.log_dir)
    
    # Training loop
    pbar = tqdm(range(start_iter, args.max_iter), desc="Training")
    for i in pbar:
        while True:
            ### DataLoader迭代器机制详解：
            # DataLoader实现了Python的迭代器协议(__iter__和__next__方法)
            # iter(dataloader): 创建一个新的迭代器对象，每次调用都会重置到数据集开头
            # next(): 从迭代器中获取下一个batch，当数据用完时会抛出StopIteration异常
            # 这种方式可以随时获取新batch，但效率不如直接for循环遍历dataloader
            image, label = next(iter(dataloader))
            if torch.unique(label).shape[0] < args.target_count:
                continue
            target = set()
            indexes = []
            for j in range(args.batch_size):
                if label[j].item() in target:
                    continue
                target.add(label[j].item())
                indexes.append(j)
                
                # 取到符合条件的target_num个数据
                if len(target) == args.target_count:
                    break
            batch_image = image[indexes]
            batch_text  = label[indexes]
            break
        
        logits = model(batch_image.to(device), batch_text.to(device))
        
        targets = torch.arange(0, args.target_count).to(device)

        # 交叉熵损失来计算loss
        loss_i = F.cross_entropy(logits, targets)
        loss_t = F.cross_entropy(logits.permute(1,0), targets)
        loss = (loss_i + loss_t) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Log loss to TensorBoard
        if i % args.log_step == 0:
            writer.add_scalar('Loss/train', loss.item(), i)
            logger.info(f"Step {i}/{args.max_iter}, Loss: {loss.item():.4f}")

        # Save model checkpoint
        if (i + 1) % args.save_step == 0:
            model_path = os.path.join(args.model_dir, f"model_step_{i + 1}.pt")
            save_checkpoint(model, optimizer, i, model_path)
            logger.info(f"Model saved at step {i + 1}")

    # Close the TensorBoard writer
    writer.close()
    logger.info("Training completed!")

def main():
    """Main function"""
    args = parse_args()
    train_model(args)

if __name__ == "__main__":
    main()

