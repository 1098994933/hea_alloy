"""
VQVAE generate samples (弃用)
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("./data/1_phase_ml_dataset.csv")
elements_col = list(dataset.columns[:-1])
print(dataset.shape)
print(elements_col)
X = dataset[elements_col]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
#print(X.head(10))
data = torch.Tensor(X)
# 设置超参数
batch_size = 64
input_dim = len(elements_col)  # 一维向量的长度
num_embeddings = 512  # 码本中的向量数量
embedding_dim = 50  # 码本向量的维度
epochs = 1000


# 定义编码器
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.relu(self.fc3(x))

# 定义量化层
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z):
        # 计算每个输入向量与码本中所有向量的距离
        flat_z = z.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_z**2, dim=1, keepdim=True)
                     - 2 * torch.matmul(flat_z, self.embeddings.weight.t())
                     + torch.sum(self.embeddings.weight**2, dim=1))
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings(encoding_indices).view(z.shape)
        return quantized

# 定义解码器
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, input_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 创建模型实例
encoder = Encoder()
quantizer = VectorQuantizer(num_embeddings, embedding_dim)
decoder = Decoder()

# 定义损失函数和优化器
optimizer = optim.Adam(list(encoder.parameters()) + list(quantizer.parameters()) + list(decoder.parameters()), lr=0.001)

# # 生成数据
# data = generate_data(500)

# 训练模型
for epoch in range(epochs):
    encoded = encoder(data)
    quantized = quantizer(encoded)
    decoded = decoder(quantized)
    loss = nn.MSELoss()(decoded, data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

# 编码一个新的数据点
# new_data = generate_data(1)
# encoded_data = encoder(new_data)

# 生成一个新的一维向量
#random_code = torch.randn(1, embedding_dim)
random_code = encoder(data[0:2])
generated_data = decoder(random_code)
print(generated_data)
print(sum(generated_data))
print(scaler.inverse_transform(generated_data.detach().numpy()))
# 可视化原始数据和生成的数据（这里仅展示部分维度以简化可视化）
# plt.scatter(data[:5, :5].detach().numpy(), np.zeros((5, 5)), label='Original Data')
# plt.scatter(generated_data[:5, :5].detach().numpy(), np.zeros((5, 5)), label='Generated Data')
# plt.legend()
# plt.show()