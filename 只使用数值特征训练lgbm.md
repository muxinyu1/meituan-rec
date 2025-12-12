# 只使用数值特征训练lgbm（CTR预估）

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

## 训练LGBM

此次训练为了探求只给LGBM数值列，研究它是否能够预测好

在数据预处理之后，需要把原数据中的类别值变为连续值，变的方法如下

- userid: 类别  -> userid_count，训练集该用户出现的次数
- itemid: 类别  -> item_count
- timestamp: 数值 -> 删除
- geohash: 类别 -> 删除
- cityid: 类别  -> cityid_count
- loc_cityid: 类别  -> 删除
- distance: 数值  
- item_ave_price: 数值  
- price: 数值  
- user_home_dis: 数值  
- user_work_dis: 数值  
- weekday: 类别 -> weekday_sin、cos
- hour: 类别 -> hour_sin、cos
- weather: 类别 -> weather_count
- temp: 数值  
- temp_low: 数值  
- temp_high: 数值  
- user_displayed_item_num: 数值
- online_days: 数值
- dtype: 类别 -> dtype_count, dtype_ctr
- cate_1: 类别  -> cate_1_count, cate_1_ctr
- cate_2: 类别  -> ...同上
- cate_3: 类别  -> ...同上
- age: 类别 -> ...同上
- level: 类别 -> ...同上
- gender: 类别  -> gender_ctr
- married: 类别 -> married_ctr
- job: 类别 -> job_ctr
- has_car: 类别 -> 删除
- work_geohash: 类别  -> 删除
- mobile_type: 类别  -> 删除
- mobile_os: 类别  -> 删除

## 训练参数：

```json
{
    "bagging_fraction": "0.9879639408647978",
    "bagging_freq": "6",
    "feature_fraction": "0.6003115063364057",
    "lambda_l1": "1.984423118582435",
    "lambda_l2": "1.234963019255433",
    "learning_rate": "0.06504878444394528",
    "max_bin": "380",
    "max_depth": "7",
    "min_child_samples": "128",
    "min_child_weight": "0.030122914019804194",
    "min_gain_to_split": "0.030592644736118974",
    "num_leaves": "61"
}
```

## 训练方法

K折交叉验证，K默认是8，划分时按照weekday和hour抽样，保证验证集的weekday和hour分布和训练集一致

## 验证集评估

AUC：每个fold的按照该fold的auc加权平均

## 预测

K折交叉预测`test.csv`，生成`submission_lgb.csv`:

```csv
sample_index,label
...,...
```