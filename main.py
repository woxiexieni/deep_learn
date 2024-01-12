import pandas as pd
import torch
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
import data_loader
from sklearn.preprocessing import StandardScaler,LabelEncoder
import classfication_net
from sklearn.ensemble import VotingClassifier #使用集成学习
import models
import My_CNN

'''
用于判断两个模型
'''
def custom_voting(outputs1, outputs2):
    are_equal = outputs1 == outputs2

    # 复制 random_choice，使其形状与 outputs1 相同
    random_choice = torch.rand(outputs1.shape).view_as(outputs1)

    # 如果相等，取 outputs1，否则取 random_choice
    ensemble_predictions = torch.where(are_equal, outputs1, random_choice)

    return ensemble_predictions


'''
MYModel_1数据准备
'''
data = data_loader.data_loader(r'D:\pythonProject\deep_learn\class_design\data\train.csv')
#删除幸存者这一列
X = data.drop(['Survived','Name','Ticket','Cabin'],axis= 1)
Y = data['Survived']
# 处理缺失值（这里简单处理，您可能需要更复杂的策略）
data = X.fillna(data.mean(numeric_only=True))  # 使用均值填充缺失值，
#对性别进行处理
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])

# 使用独热编码处理 "Embarked" 列
data = pd.get_dummies(data, columns=['Embarked'], prefix='Embarked')
print("数据形状：", data.shape)
# 划分训练集和测试集
# 例如，对于数值特征，可以使用标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# 转换为 PyTorch 的 Tensor
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)  # 转换为列向量
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)  # 转换为列向量
# 3. 实例化模型和损失函数、优化器
input_size = X_train.shape[1]
model_1 = classfication_net.MYModel_1(input_size)
#二元交叉损失函数
criterion = nn.BCELoss()

#使用 Adam 优化器来更新神经网络模型的参数
optimizer = optim.Adam(model_1.parameters(), lr=0.001)
# 4. 训练模型
epochs = 800
for epoch in range(epochs):
    model_1.train()
    optimizer.zero_grad()
    outputs = model_1(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
torch.save(model_1.state_dict(), 'model_1_weights.pth')
# 5. 评估模型
#model_1.load_state_dict(torch.load('model_1_weights.pth'))
model_1.eval()
with torch.no_grad():
    outputs = model_1(X_test_tensor)
    predictions = (outputs >= 0.5).float()
    accuracy = (predictions == y_test_tensor).float().mean()
    print(f'Accuracy on test set: {accuracy.item()}')

'''
预测
'''

# 预测集数据准备
data_test = data_loader.data_loader(r'D:\pythonProject\deep_learn\class_design\data\test.csv')

# 删除 'Name'、'Ticket'、'Cabin' 列
X_test_data = data_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# 使用与训练集相同的方式处理预测集数据
X_test_data = X_test_data.fillna(X_test_data.mean(numeric_only=True))
X_test_data['Sex'] = le.transform(X_test_data['Sex'])
X_test_data = pd.get_dummies(X_test_data, columns=['Embarked'], prefix='Embarked')
X_test_scaled = scaler.transform(X_test_data)

# 转换为 PyTorch 的 Tensor
X_test_tensor_1 = torch.FloatTensor(X_test_scaled)

with torch.no_grad():
    prediction = model_1(X_test_tensor_1)

    predicted_class = (prediction >= 0.5).float()
    #print(f'Predicted class: {predicted_class}')
    results_df = pd.DataFrame(
        {'PassengerId': data_test['PassengerId'].values, 'Survived': predicted_class.numpy().flatten().astype(int)})

    # 将 DataFrame 写入 CSV 文件
    results_df.to_csv('test_results.csv', index=False)
    print('全连接网络训练结束')


'''
CNN网络训练
'''
model = My_CNN.My_Cnn()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 800
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# 评估模型
model.eval()
torch.save(model.state_dict(), 'model_weights.pth')
with torch.no_grad():
    outputs = model(X_test_tensor)
    predictions = (outputs >= 0.5).float()
    accuracy = (predictions == y_test_tensor).float().mean()
    print(f'Accuracy on test set: {accuracy.item()}')

'''
使用CNN模型进行预测
'''
with torch.no_grad():
    prediction = model(X_test_tensor_1)

    predicted_class = (prediction >= 0.5).float()
    # print(f'Predicted class: {predicted_class}')
    results_df = pd.DataFrame(
        {'PassengerId': data_test['PassengerId'].values, 'Survived': predicted_class.numpy().flatten().astype(int)})

    # 将 DataFrame 写入 CSV 文件
    results_df.to_csv('test01_results.csv', index=False)
    print('CNN网络训练结束')

'''使用集成学习'''
# 创建 EnsembleClassifier
# 创建自定义模型的实例
model1 = classfication_net.MYModel_1(input_size)# 假设 input_size 是你的模型的输入大小
model2 = My_CNN.My_Cnn()
# 加载模型的权重
model1.load_state_dict(torch.load('model_1_weights.pth'))
model2.load_state_dict(torch.load('model_weights.pth'))

# 在测试集上进行预测
with torch.no_grad():
    # 假设 X_test_tensor 是你的测试数据的 PyTorch Tensor
    outputs1 = (model1(X_test_tensor_1) >= 0.5).float()

    outputs2 = (model2(X_test_tensor_1) >= 0.5).float()
# 进行投票
ensemble_predictions = torch.round(custom_voting(outputs1, outputs2)).view(-1).int()
# ensemble_predictions = torch.floor((W[0]*outputs1 + W[1]*outputs2 ) / 2.0).view(-1)
# print(f'1111:{ensemble_predictions}')
# 将投票结果转换为 numpy 数组
ensemble_predictions_numpy = ensemble_predictions.numpy()

results_df = pd.DataFrame({'PassengerId': data_test['PassengerId'].values,
                            'Survived': ensemble_predictions})

# 将 DataFrame 写入 CSV 文件
results_df.to_csv('my_results.csv', index=False)











