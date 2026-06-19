import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
# 假设你的注意力矩阵形状是 [23, 23, 2]
# 你可以替换这行为真实数据加载
attn = torch.load('drug_att_597.pt').cpu().numpy()

#attn = np.random.rand(23, 23, 2)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 2 heads

for i in range(2):
    sns.heatmap(attn[:, :, i], ax=axs[i], cmap='YlGnBu', square=True, cbar=True)
    axs[i].set_title(f'Attention Head {i+1}', fontsize=14)
    axs[i].set_xlabel('Key', fontsize=12)
    axs[i].set_ylabel('Query', fontsize=12)

plt.tight_layout()
plt.show()