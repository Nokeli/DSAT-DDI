import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 假设数据格式：
# drug_vectors: (1052, 80) 的numpy数组或torch tensor
# drug_labels: 长度为1052的列表，包含每个药物的唯一标签

# 示例数据生成（替换为您的实际数据）
drug_vectors = torch.load('drug1_re.pt').cpu().numpy()
drug_labels = [i for i in range(1052)]  # 您的1052个药物标签

# 标准化数据
scaler = StandardScaler()
drug_vectors_scaled = scaler.fit_transform(drug_vectors)

# 为每个标签生成唯一颜色
unique_labels = list(set(drug_labels))
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
label_to_color = dict(zip(unique_labels, colors))
n_col = 5 if len(unique_labels) > 30 else 1
# 先使用PCA降维加速t-SNE
pca_50 = PCA(n_components=50)
drug_vectors_pca = pca_50.fit_transform(drug_vectors_scaled)

# 自动确定perplexity
perplexity = min(30, (len(drug_vectors_pca) - 1) // 3)

# t-SNE降维
tsne = TSNE(n_components=2, perplexity=perplexity,
            random_state=42, n_iter=1000)
tsne_results = tsne.fit_transform(drug_vectors_pca)

plt.figure(figsize=(16, 10))
for label, color in label_to_color.items():
    mask = np.array(drug_labels) == label
    plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                color=color, label=label, alpha=0.7, s=50)

plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title(f'1052个药物表征的t-SNE可视化 (perplexity={perplexity}, 按标签着色)')

# 图例处理
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=n_col)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()