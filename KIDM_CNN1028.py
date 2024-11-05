# -*- coding: utf-8 -*-
from torch import optim

import torch

import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.io as scio
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd
import torch.nn.utils.prune as prune
from sklearn.metrics import r2_score#R square
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import gaussian_filter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.backends.cudnn.deterministic = True  # 设置GPU计算为确定性

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def smape(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / (0.5*(np.abs(y_pred) + np.abs(y_true)))) * 100



class DeepKoopmanControl(nn.Module):
    def __init__(self):
        super(DeepKoopmanControl, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3,stride=2, padding=2),
            nn.SELU(inplace=True),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3,stride=2, padding=3),
            nn.SELU(inplace=True),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,stride=2, padding=3),
            nn.SELU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,stride=2, padding=3),
            nn.SELU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=1, kernel_size=3,stride=2, padding=3)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=1, out_channels=8, kernel_size=2, stride=2, padding=2),
            nn.SELU(inplace=True),
            nn.ConvTranspose1d(in_channels=8, out_channels=16, kernel_size=2, stride=2, padding=3),
            nn.SELU(inplace=True),
            nn.ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=2, stride=2, padding=3),
            nn.SELU(inplace=True),
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=3),
            nn.SELU(inplace=True),
            nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=2, stride=2, padding=4)
        )
        # self.decoder = nn.Linear(12, 100)

        # self.K = nn.Linear(4, 64, bias=False)
        self.K = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.SELU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=8, kernel_size=3, stride=1, padding=2)
        )



        self.lr = nn.Linear(80, 1, bias=True)


    def forward(self, x0, u):
        #print(x0.shape)
        xx = torch.cat((x0,u), dim=1)
        xx = xx.unsqueeze(1)

        #xx = torch.cat((x0,u), dim=1)
        yk0 = self.encoder(xx)
        yk0 = yk0.view(-1, 8, 1) 
        #print(yk0.shape)
        Ku = self.K(u.unsqueeze(1))
        #Ku = Ku.view(-1,8,8)
        #print(Ku.shape)
        yk1 = torch.matmul(Ku, yk0)
        #print(yk1.shape)
        yk1 = yk1.view(-1,1,8)
        yk0 = yk0.view(-1,1,8)
        #xb1 = self.B2(u)


        x1 = self.decoder(yk1)
        #print(x1.shape)
        x0_hat = self.decoder(yk0)

        return x1.view(-1,100), x0_hat.view(-1,100), yk0.view(-1,8), yk1.view(-1,8)


model=DeepKoopmanControl().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-3)
#optimizer = optim.SGD(model.parameters(), lr=0.0001)
criterion = nn.MSELoss().to(device)
#criterion = nn.SmoothL1Loss().to(device)
datax=scio.loadmat('traindata.mat')
datay2=scio.loadmat('trainrul.mat')
train_X=datax['Dtr']
train_Y2=datay2['rultr']
train_wc=scio.loadmat('trainu.mat')
train_wc=train_wc['utr']
dataset1 = train_X
num=len(dataset1[0])
img=np.ones((num,1000))
for i in range(num):
    img[i,:]=dataset1[0,i][:,1]
#img=img.reshape(num,2,1000)
img = torch.FloatTensor(img).to(device)
train_wc = torch.FloatTensor(train_wc/5).to(device)

rul0=torch.FloatTensor(train_Y2.T).to(device)
train_losses = []
train_dataset = TensorDataset(img, train_wc, rul0)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

valx1=scio.loadmat('data4.mat')
valy1=scio.loadmat('4RUL0.mat')
val_X1=valx1['data']
val_Y1=valy1['rul']
valx2=scio.loadmat('data72.mat')
valy2=scio.loadmat('72RUL0.mat')
val_X2=valx2['data']
val_Y2=valy2['rul']
df = pd.read_csv('wc.csv',header=None)
wc = df.to_numpy()
val_wc1=wc[4]
val_wc2=wc[71]
numval1=len(val_X1[0])
imgval1=np.ones((numval1,1000))
numval2=len(val_X2[0])
imgval2=np.ones((numval2,1000))
for j in range(numval1):
    imgval1[j,:]=val_X1[0,j][:,1]
for k in range(numval2):
    imgval2[k,:]=val_X2[0,k][:,1]
imgval = np.vstack((imgval1, imgval2))
val_wc = np.vstack((val_wc1,val_wc2))
#img=img.reshape(num,2,1000)
imgval = torch.FloatTensor(imgval).to(device)
val_wc1 = torch.FloatTensor(val_wc1/5).to(device)
val_wc2 = torch.FloatTensor(val_wc2/5).to(device)
val_wc = torch.cat((val_wc1.repeat(numval1,1),val_wc2.repeat(numval2,1)),dim=0)
val_Y = np.hstack((val_Y1, val_Y2))
rulval=torch.FloatTensor(val_Y.T).to(device)
# 早停参数
patience = 40
best_loss = float('inf')
counter = 0

num_epochs =600
for step in range(num_epochs):
    model.train()
    print(step)
    loss = 0.0
    for xx, uu, RUL in train_loader:
        segments = torch.split(xx, 100, dim=1)
        observables = []
        loss_rec = 0
        loss_lin = 0
        loss_pred = 0
        i = 0
        _, x0_hat, _, _ = model(segments[0],uu)
        loss_rec = criterion(segments[0], x0_hat)
        for i in range(len(segments)-1):
            xtm,_,yk0,yktm = model(segments[i],uu)
            observables.append(yk0)
            _,_,yk1,_ = model(segments[i+1],uu)
            loss_lin += criterion(yk1,yktm)
            loss_pred += criterion(segments[i+1],xtm)
        _, _, yk9, _ = model(segments[9],uu)
        observables.append(yk9)
        output = torch.cat(observables, dim=1)
        rul_hat = model.lr(output)
        #loss_rul = criterion(rul_hat,RUL)+0.01 * torch.norm(model.lr.weight, p=1)
        loss_rul = criterion(rul_hat,RUL)
        loss = loss_rec+loss_lin+loss_pred+loss_rul
        optimizer.zero_grad()
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
    model.eval()
    val_loss = 0
    with torch.no_grad():
        segmentsval = torch.split(imgval, 100, dim=1)
        observablesval = []
        for segval in segmentsval:
            _,_,obs,_ = model(segval,val_wc)
            observablesval.append(obs)
        output_val = torch.cat(observablesval, dim=1)
        rul_val = model.lr(output_val)
        rul_val = gaussian_filter(rul_val.cpu(), sigma=2)
        loss_val = criterion(torch.Tensor(rul_val),rulval.cpu())
        val_loss = loss_val.item()
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping triggered at epoch {step+1}')
                break

##test


qq=[4,7,13,14,19,23,24,25,29,31,35,37,38,45,46,49,52,53,61,68,71,72]
df = pd.read_csv('wc.csv',header=None)
wc = df.to_numpy()
model.eval()
with torch.no_grad():
    for t in range(22):
        test=scio.loadmat('data'+str(qq[t])+'.mat')
        testy2=scio.loadmat(str(qq[t])+'RUL0.mat')
        test_X=test['data']
        test_Y2=testy2['rul']
        test_wc=wc[qq[t]-1]
        num2=len(test_X[0])
        img2=np.ones((num2,1000))
        for j in range(num2):
            img2[j,:]=test_X[0,j][:,1]
        #img=img.reshape(num,2,1000)
        img2 = torch.FloatTensor(img2).to(device)
        test_wc = torch.FloatTensor(test_wc/5).to(device)
        rulte=torch.FloatTensor(test_Y2.T).to(device)
        segments = torch.split(img2, 100, dim=1)
        observables = []
        for segment in segments:
            _,_,observable,_ = model(segment,test_wc.repeat(num2,1))
            observables.append(observable)
        output = torch.cat(observables, dim=1)
        rul_hat = model.lr(output)
        rul_hat = rul_hat.view(1,-1)
        rulpr=rul_hat.cpu().detach().numpy()
        rulpr0=rulpr
        rultx=np.array(rulpr0)
        rultx=gaussian_filter(rultx, sigma=2)
        rultest=test_Y2
        MSE = mean_squared_error(rultest, rultx)
        RMSE = (mean_squared_error(rultest, rultx))**0.5
        MAE = mean_absolute_error(rultest, rultx)
        MAPE = mape(rultest, rultx)
        SMAPE = smape(rultest, rultx)
        RSquared=r2_score(rultest.T, rultx.T)
        print("MSE:", MSE)
        print("RMSE:", RMSE)
        print("MAE:", MAE)
        print("MAPE:", MAPE)
        print("SMAPE:", SMAPE)
        print("RSquared:", RSquared)
        plt.plot(rultest.T, color='blue', label='y0')
        
        # 绘制y2序列的曲线，颜色为红色
        plt.plot(rultx.T, color='red', label='y2')
        plt.show()  # 显示图形

        SM2=pd.DataFrame(rultx)
        SM2.to_csv(str(qq[t])+'YCRUL3.csv')