import ssl
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from torchvision.models import resnet50, ResNet50_Weights

client = OpenAI(api_key='sk-proj-XejKLwxIzRbmN7aZKq86T3BlbkFJ16IZfyRldru3gAOqTOa4')  # 替换为您的实际API密钥
ssl._create_default_https_context = ssl._create_unverified_context
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

def generate_and_analyze_samples(model, sample_size=100, num_groups=100):
    layers = [module for module in model.modules() if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))]
    all_trained_samples = []
    all_random_samples = []

    for group in range(num_groups):
        selected_layer = random.choice(layers)
        trained_weights = selected_layer.weight.data
        random_weights = torch.randn_like(trained_weights)

        trained_sample = trained_weights.flatten()[:sample_size].numpy()
        random_sample = random_weights.flatten()[:sample_size].numpy()

        all_trained_samples.extend(trained_sample)
        all_random_samples.extend(random_sample)

    # 将所有样本合并成一个数组
    all_trained_samples = np.array(all_trained_samples)
    all_random_samples = np.array(all_random_samples)

    # 创建直方图
    plt.figure(figsize=(12, 6))

    # 训练样本的直方图
    plt.subplot(1, 2, 1)
    plt.hist(all_trained_samples, bins=50, alpha=0.7, color='blue')
    plt.title('Trained Weights Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')

    # 随机样本的直方图
    plt.subplot(1, 2, 2)
    plt.hist(all_random_samples, bins=50, alpha=0.7, color='red')
    plt.title('Random Weights Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('weight_distributions.png')
    plt.show()

generate_and_analyze_samples(model, sample_size=100, num_groups=100)
