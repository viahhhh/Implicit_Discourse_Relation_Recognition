import numpy as np
import time
import tqdm
import load_data as ld
import torch
import torch.nn as nn
from model import Bi_LSTMmodel,Transformer_model
from torch.utils.data import  DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

max_sent_len = 256    #一个句子最大长多少
batch_size = 16       #batch大小
epochs = 10            #重复次数
lr = 5e-6             #学习率

print("loading dataset...")
dataset = ld.Corpus(max_sent_len)
print("dataset loading completed")

#设定模型与loss_function,optimizer
#model = Bi_LSTMmodel(vocab_size,embedding_dim,hidden_size,num_layers,max_sent_len).to(device)
model = Transformer_model().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

#加载数据集
train_set = DataLoader(dataset=dataset.train,batch_size=batch_size,shuffle=False)
test_set = DataLoader(dataset=dataset.test,batch_size=batch_size,shuffle=False)
vaild_set = DataLoader(dataset=dataset.vaild,batch_size=batch_size,shuffle=False)


def valid():
    '''
    进行验证，返回模型在验证集上的 accuracy
    '''
    sum_true = 0

    model.eval()
    with torch.no_grad():
        for data in vaild_set:
            # 输入arg1与arg2与label
            batch = data[0].to(device)
            batch_mask = data[1].to(device)
            batch_y = data[2].to(device)

            # 预测
            y_hat = model(batch,batch_mask)
            # 取分类概率最大的类别作为预测的类别
            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)
            for i in range(y_hat.shape[0]):
                if y_hat[i] in batch_y[i]:
                    sum_true += 1

        return sum_true / dataset.vaild.__len__()


def test():
    # 测试
    TP = [0., 0., 0., 0.]  # 真正样本
    FN = [0., 0., 0., 0.]  # 假反例
    FP = [0., 0., 0., 0.]  # 假正例
    model.eval()
    with torch.no_grad():
        for data1 in tqdm.tqdm(test_set):
            # 输入arg1与arg2与label
            batch1 = data1[0].to(device)
            batch_mask1 = data1[1].to(device)
            batch_y1 = data1[2].to(device)

            # 预测
            y_hat1 = model(batch1, batch_mask1)
            y_hat1 = torch.tensor([torch.argmax(_) for _ in y_hat1]).to(device)
            print(y_hat1.shape[0])
            for i in range(y_hat1.shape[0]):
                for j in range(4):
                    if j in batch_y1[i]:  # 代表batch_y[i]为j的真实正样本
                        if j == y_hat1[i]:  # 代表y_hat[i]为j的预测正样本
                            TP[j] += 1
                        else:  # 代表预测为负样本
                            FN[j] += 1
                    else:  # 代表batch_y[i]为j的真实负样本
                        if j == y_hat1[i]:  # 代表y_hat[i]为j的预测正样本
                            FP[j] += 1
    F1_sum = 0.
    print(TP)
    print(FP)
    print(FN)
    for i in range(4):
        P = TP[i] / (TP[i] + FP[i])
        R = TP[i] / (TP[i] + FN[i])
        F1_sum += (2 * P * R) / (P + R)
    Marco_F1 = F1_sum / 4
    print(f"测试集上Marco_F1的值为:{Marco_F1 * 100:.2f}%")

#训练
max_valid_acc = 0
for epoch in range(epochs):
    sum_true = 0
    sum_loss = 0.0

    model.train()
    for data in tqdm.tqdm(train_set):
        #输入arg1与arg2与label
        batch = data[0].to(device)
        batch_mask = data[1].to(device)
        #因为训练集都只有一个标签，所以只需要取第一个就好了
        batch_y = data[2][:,0].to(device)
        #预测
        y_hat = model(batch,batch_mask)

        #进行反向传播
        loss = loss_function(y_hat, batch_y.to(torch.int64))
        loss.backward()         # 计算梯度
        optimizer.step()        # 更新参数
        optimizer.zero_grad()

        #计算一些loss与正确率
        y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)
        sum_true += torch.sum(y_hat == batch_y).float()
        sum_loss += loss.item()

    train_acc = sum_true / dataset.train.__len__()
    train_loss = sum_loss / (dataset.train.__len__() / batch_size)
    #验证
    valid_acc = valid()

    print(f"epoch: {epoch}, train loss: {train_loss:.4f}, train accuracy: {train_acc*100:.2f}%, valid accuracy: {valid_acc*100:.2f}%,\
            time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) }")
    test()

