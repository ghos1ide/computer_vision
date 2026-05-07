import jittor as jt
from jittor import nn
from jittor.dataset.cifar import CIFAR10
from jittor import transform
import matplotlib.pyplot as plt
from model import SimpleResNet

# 开启GPU加速（如果有的话），Jittor会自动使用CUDA
jt.flags.use_cuda = jt.has_cuda

def train(model, train_loader, optimizer, epoch, train_losses):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = nn.cross_entropy_loss(output, target)
        optimizer.step(loss)
        
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}] \tLoss: {loss.item():.6f}")
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    return avg_loss

def test(model, test_loader, test_accs):
    model.eval()
    correct = 0
    total = 0
    for data, target in test_loader:
        output = model(data)
        pred = jt.argmax(output, dim=1)[0]
        correct += (pred == target).sum().item()
        total += target.shape[0]
        
    acc = correct / total
    print(f"Test set: Accuracy: {correct}/{total} ({acc:.4f})")
    test_accs.append(acc)
    return acc

def plot_metrics(train_losses, test_accs, title_info=""):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    if title_info:
        plt.suptitle(title_info)
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accs, 'r-', label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig('training_metrics.png')
    plt.show()

def main():
    # 1. 数据集准备
    # 使用 CIFAR-10 数据集
    transform_train = transform.Compose([
        transform.RandomCrop(32),
        transform.RandomHorizontalFlip(),
        transform.ImageNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    transform_test = transform.Compose([
        transform.ImageNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    
    # 可调超参数
    batch_size = 16  # 可根据需要修改
    learning_rate = 0.003
    epochs = 100

    train_loader = CIFAR10(train=True, transform=transform_train).set_attrs(batch_size=batch_size, shuffle=True)
    test_loader = CIFAR10(train=False, transform=transform_test).set_attrs(batch_size=batch_size, shuffle=False)

    # 2. 初始化模型
    model = SimpleResNet(num_classes=10, dropout_prob=0.3)

    # 3. 设置优化器
    optimizer = nn.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    train_losses = []
    test_accs = []

    # 4. 开始训练
    print("开始训练...")
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, epoch, train_losses)
        test(model, test_loader, test_accs)

    print("训练结束！保存指标图...")
    title_info = f"Hyperparameters: LR={learning_rate}, Epochs={epochs}, Batch Size={batch_size}, Optimizer=SGD"
    plot_metrics(train_losses, test_accs, title_info)

if __name__ == '__main__':
    main()
