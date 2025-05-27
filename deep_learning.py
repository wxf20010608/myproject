import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import mean_absolute_error
# 处理 notRepairedDamage 为 '-' 的数据（替换为 0.5 并转为 float）
df = pd.read_csv(r'D:\something_to_test\used_car_train_20200313.csv', sep=' ')
df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', 0.5).astype(float)

data_test = pd.read_csv(r'D:\something_to_test\used_car_testB_20200421.csv', sep=' ')
data_test['notRepairedDamage'] = data_test['notRepairedDamage'].replace('-', 0.5).astype(float)
df.shape
# 预处理
def date_proc_zero(x):
    m = int(x[4:6])
    if m == 0:
        m = 1
    return x[:4] + '-' + str(m) + '-' + x[6:]
def parse_date(df, colname):
    newcol = colname + 'timestamp'
    df[newcol] = pd.to_datetime(df[colname].astype('str').apply(date_proc_zero))
    df[colname + '_year'] = df[newcol].dt.year
    df[colname + '_month'] = df[newcol].dt.month
    df[colname + '_day'] = df[newcol].dt.day
    df[colname + '_dayofweek'] = df[newcol].dt.dayofweek
    return df
train_data = df
train_data = parse_date(train_data, 'regDate')
train_data = parse_date(train_data, 'creatDate')
# 构造特征--计算车龄，以月为单位
train_data['carAge'] = (train_data['creatDate_year'] - train_data['regDate_year']) * 12 + train_data['creatDate_month'] - train_data['regDate_month']
  
data_test = parse_date(data_test, 'regDate')
data_test = parse_date(data_test, 'creatDate')
# 构造特征--计算车龄，以月为单位
data_test['carAge'] = (data_test['creatDate_year'] - data_test['regDate_year']) * 12 + data_test['creatDate_month'] - data_test['regDate_month']
train_data.info()
#修改异常数据
train_data.loc[train_data['power'] > 600, 'power'] = 600  # 正确写法
data_test.loc[data_test['power'] > 600, 'power'] = 600    # 正确写法
train_data = pd.get_dummies(train_data, prefix=None, prefix_sep='_', dummy_na=False, columns=['model','bodyType','gearbox','brand','fuelType','notRepairedDamage'], sparse=False, drop_first=False)
train_data.head()
data_test = pd.get_dummies(data_test, prefix=None, prefix_sep='_', dummy_na=False, columns=['model','bodyType','gearbox','brand','fuelType','notRepairedDamage'], sparse=False, drop_first=False)
data_test.head()
missing_cols = set( train_data.columns ) - set( data_test.columns )
print(missing_cols)
data_test['model_247.0'] = 0
train_data.fillna(train_data.median(),inplace= True)
data_test.fillna(train_data.median(),inplace= True)
tags=list(train_data.columns)
print(len(tags))
print(tags)
tags.remove('price')
tags.remove('creatDatetimestamp')
tags.remove('SaleID')
tags.remove('regDatetimestamp')
print(len(tags))
print(tags)
# 假设 train_data 和 data_test 都经过了 get_dummies 或类似的操作
train_data, data_test = train_data.align(data_test, join='left', axis=1, fill_value=0)
#特征归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(train_data[tags].values)
x = min_max_scaler.transform(train_data[tags].values)
x_ = min_max_scaler.transform(data_test[tags].values)
#获得y值
y = train_data['price'].values
#切分训练集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1)
from tensorflow.keras import regularizers
model = keras.Sequential([
    keras.layers.Dense(400, activation='relu', kernel_regularizer=regularizers.l2(0.01)), 
    keras.layers.Dense(200, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  
    keras.layers.Dense(200, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(1)
])
model.compile(loss='mean_absolute_error',
                optimizer='Adam')
model.fit(x_train,y_train,batch_size = 2048,epochs=500)   # 100+10+10+10
#比较训练集和验证集效果
print(mean_absolute_error(y_train,model.predict(x_train)))
test_pre = model.predict(x_test)
print(mean_absolute_error(y_test,test_pre))#输出结果预测
y_=model.predict(x_)
# 保存模型
data_test_price = pd.DataFrame(y_,columns = ['price'])
results = pd.concat([data_test['SaleID'],data_test_price],axis = 1)
results.to_csv('results.csv',sep = ',',index = None)
