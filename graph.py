import numpy as np
from matplotlib import pyplot as plt
import copy
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

items = np.load('data/xydata/raw/data.npz')
x = items['x']
y = items['y']
edge_index = items['edge_index']
edge_type = items['edge_type']
train_mask = items['train_mask']
test_mask = items['test_mask']
edge_timestampe = items['edge_timestamp']
print(x.shape)
print(y.shape)
print(edge_index.shape)
print(edge_type.shape)
print(train_mask.shape)
print(edge_timestampe.shape)

y_his = copy.deepcopy(y)
for i in range(len(y_his)):
    if y_his[i] == -100:
        y_his[i] = 1
y_his = np.histogram(y_his, bins = [-1,0.5,1.5,2.5,3.5])
print(y_his[0])
#bar_x = [0,1,2,3]
'''
bar_x = ['正常用户(Class 0)','欺诈用户(Class 1)','无关用户(Class 2)','无关用户(Class 3)']
plt.bar(bar_x, y_his[0], align =  'center')
plt.title('四类节点数量直方图')
plt.ylabel('用户数量')
plt.xlabel('用户分类')
#plt.show()
'''
'''
num = 0
sum = 0
for i in range(17):
    for j in range(len(x)):
        sum += 1
        if x[j][i] == -1:
            num+=1
print(num,sum,num/sum)

num = 0
sum = 0
for i in range(17):
    for j in range(len(x)):

        if y[j] == 0:
            sum += 1

            if x[j][i] == -1:
                num+=1
print(num,sum,num/sum)

num = 0
sum = 0
for i in range(17):
    for j in range(len(x)):
        if y[j] == -100:
            sum += 1
            if x[j][i] == -1:
                num+=1
print(num,sum,num/sum)

num = 0
sum = 0
for i in range(17):
    for j in range(len(x)):
        if y[j] == 2:
            sum += 1
            if x[j][i] == -1:
                num+=1
print(num,sum/17,num/sum)
num = 0
sum = 0
for i in range(17):
    for j in range(len(x)):
        if y[j] == 3:
            sum += 1
            if x[j][i] == -1:
                num+=1
print(num,sum/17,num/sum)
plt.hist(edge_timestampe,bins=[0,100,200,300,400,500])
plt.show()
'''
rudu = np.zeros(3704457)
chudu = np.zeros(3704457)

for i in range(len(edge_index)):
    rudu[edge_index[i][1]] += 1
    chudu[edge_index[i][0]] += 1

print(np.histogram(rudu,bins=[-1,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,100]))
print(np.histogram(chudu,bins=[-1,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,100]))

rudu = np.zeros(3704457)
chudu = np.zeros(3704457)

for i in range(len(edge_index)):
    rudu[edge_index[i][1]] += 1
    chudu[edge_index[i][0]] += 1
for i in range(len(y)):
    if y[i] != 0:
        rudu[i] = 0
        chudu[i] = 0
print(np.histogram(rudu,bins=[-1,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,100]))
print(np.histogram(chudu,bins=[-1,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,100]))

rudu = np.zeros(3704457)
chudu = np.zeros(3704457)

for i in range(len(edge_index)):
    rudu[edge_index[i][1]] += 1
    chudu[edge_index[i][0]] += 1
for i in range(len(y)):
    if y[i] != -100:
        rudu[i] = 0
        chudu[i] = 0
print(np.histogram(rudu,bins=[-1,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,100]))
print(np.histogram(chudu,bins=[-1,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,100]))
