# 只使用类别特征训练nn（CTR预估）

## 训练文件

`train.csv`:
- label: 是否点击（0、1）
- userid: 类别  
- itemid: 类别  
- timestamp: 数值
- geohash: 类别
- cityid: 类别  
- loc_cityid: 类别  
- distance: 数值  
- item_ave_price: 数值  
- price: 数值  
- user_home_dis: 数值  
- user_work_dis: 数值  
- weekday: 类别
- hour: 类别
- weather: 类别
- temp: 数值  
- temp_low: 数值  
- temp_high: 数值  
- user_displayed_item_num: 数值
- online_days: 数值
- dtype: 类别
- cate_1: 类别  
- cate_2: 类别  
- cate_3: 类别  
- age: 类别
- level: 类别
- gender: 类别  
- married: 类别
- job: 类别
- has_car: 类别
- work_geohash: 类别  
- mobile_type: 类别  
- mobile_os: 类别  

`test.csv`:

结构同`train.csv`，只不过多了一个`sample_index`列

## 数据预处理

- 填充nan值
> 按照整个train.csv缺失率填充，缺失率高于10%添加一列: xxx_is_missing，表示是否缺失
> 缺失的值对于类别列，使用众数填充；对于数值列，使用中位数填充

## 训练nn

此次训练为了探求只给nn类别列，研究它是否能够预测好

在数据预处理之后，需要做一些数据处理

- userid: 类别  -> 删除
- itemid: 类别  -> 出现次数小于10的itemid划分为UNK
- timestamp: 数值 -> 删除
- geohash: 类别 -> 删除
- cityid: 类别  -> 出现次数小于200的cityid划分为UNK
- loc_cityid: 类别 -> 删除
- is_same_city: (cityid == loc_cityid)
- distance: 数值 -> 删除
- item_ave_price: 数值  -> 删除
- price: 数值  -> 删除
- user_home_dis: 数值  -> 删除
- user_work_dis: 数值  -> 删除
- weekday: 类别
- hour: 类别 
- weather: 类别
- is_weekend: weekday in (6, 7)
- timeslot: 对hour做划分
- temp: 数值  -> 17 ~ 26 为舒适温度，划分为一类，其他数值一类
- temp_low: 数值  -> 删除
- temp_high: 数值  -> 删除
- user_displayed_item_num: 数值 -> 删除
- online_days: 数值 -> 删除
- dtype: 类别 
- cate_1: 类别  
- cate_2: 类别  
- cate_3: 类别  
- age: 类别 
- level: 类别  
- gender: 类别   
- married: 类别  
- job: 类别  
- has_car: 类别  
- work_geohash: 类别  
- mobile_type: 类别  
- mobile_os: 类别 

## 模型

- 首先采用DCN V2

## 训练方法

K折交叉验证，K默认是10，划分时按照weekday和hour抽样，保证验证集的weekday和hour分布和训练集一致

## 验证集评估

AUC：每个fold的按照该fold的auc加权平均

## 预测

K折交叉预测`test.csv`，生成`submission_nn.csv`:

```csv
sample_index,label
...,...
```