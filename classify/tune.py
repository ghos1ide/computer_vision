import jittor as jt
from jittor import nn
from jittor.dataset.cifar import CIFAR10
from jittor import transform
import matplotlib.pyplot as plt
from model import SimpleResNet
from train import train, test

# 开启GPU加速（如果有的话），Jittor会自动使用CUDA
jt.flags.use_cuda = jt.has_cuda

def run_experiment(lr, batch_size, epochs):
    """
    运行单次实验的函数，接收不同的超参数
    """
    transform_train = transform.Compose([
        transform.RandomCrop(32),
        transform.RandomHorizontalFlip(),
        transform.ImageNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    transform_test = transform.Compose([
        transform.ImageNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    
    train_loader = CIFAR10(train=True, transform=transform_train).set_attrs(batch_size=batch_size, shuffle=True)
    test_loader = CIFAR10(train=False, transform=transform_test).set_attrs(batch_size=batch_size, shuffle=False)
    
    model = SimpleResNet(num_classes=10, dropout_prob=0.3)
    optimizer = nn.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
    train_losses = []
    test_accs = []
    
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, epoch, train_losses)
        test(model, test_loader, test_accs)
        
    final_acc = test_accs[-1] if test_accs else 0
    return train_losses, test_accs, final_acc

def main():
    # 定义要测试的超参数选项
    learning_rates = [0.03, 0.01, 0.003]
    batch_sizes = [16, 32, 64]
    epochs = 40 # 为了加速测试，可以先设置较少的epoch数量
    
    results = {}
    
    plt.figure(figsize=(16, 7))
    plt.suptitle("Hyperparameter Tuning Results")

    for lr in learning_rates:
        for bs in batch_sizes:
            exp_name = f"LR={lr}_BS={bs}"
            print(f"\n=========================================")
            print(f"开始测试配置: {exp_name}")
            print(f"=========================================")
            
            train_losses, test_accs, final_acc = run_experiment(lr, bs, epochs)
            results[exp_name] = final_acc
            
            # 绘制损失曲线
            plt.subplot(1, 2, 1)
            plt.plot(range(1, epochs + 1), train_losses, label=exp_name)
            
            # 绘制准确率曲线
            plt.subplot(1, 2, 2)
            plt.plot(range(1, epochs + 1), test_accs, label=exp_name)

    # 完善图表信息
    plt.subplot(1, 2, 1)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    plt.subplot(1, 2, 2)
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.savefig('hyperparam_tuning_results.png', bbox_inches='tight')
    plt.show()

    # 打印最终所有配置的对比结果
    print("\n========= 超参数测试最终结果 =========")
    # 按准确率从高到低排序
    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    for exp, acc in sorted_results:
        print(f"配置: {exp:<20} | 最终准确率: {acc:.4f}")

if __name__ == '__main__':
    main()
