# 首页推荐排序CTR建模 - CatBoost版本迭代分析

本文档详细分析了从 V1 到 V4 四个版本的 CatBoost CTR 建模脚本的迭代升级过程，包括每个版本新增的功能及其对模型效果的影响分析。

---

## 目录
1. [V1 基础版概述](#v1-基础版概述)
2. [V2 升级内容分析](#v2-升级内容分析v1--v2)
3. [V3 升级内容分析](#v3-升级内容分析v2--v3)
4. [V4 升级内容分析](#v4-升级内容分析v3--v4)
5. [V4 vs V1 功能总结](#v4-vs-v1-全部新增功能总结)
6. [版本对比总表](#版本对比总表)

---

## V1 基础版概述

**文件**: `train_catboost.py`

V1 是基础版本，包含以下核心功能：

### 特征工程
| 特征类型 | 具体内容 |
|---------|---------|
| Count Encoding | `userid`, `itemid`, `geohash`, `cityid`, `loc_cityid` 的出现次数统计 |
| 时间特征 | `hour_sin`, `hour_cos`, `weekday_sin`, `weekday_cos` (周期性编码) |
| 类别特征 | 21个原始类别特征直接使用 |

### 模型配置
```python
params = {
    'iterations': 15000,
    'learning_rate': 0.06260215315409043,
    'depth': 10,
    'l2_leaf_reg': 3.6916543587180826,
    'boosting_type': 'Ordered',
    'one_hot_max_size': 2,
    'od_wait': 500,
    'auto_class_weights': 'Balanced',
}
```

### 其他特性
- 内存优化 (`reduce_mem_usage`)
- 10折交叉验证（按 weekday 分层抽样）
- 简单平均集成预测
- GPU加速训练

---

## V2 升级内容分析（V1 → V2）

**文件**: `train_catboost_v2.py`

### 新增功能一览

| 序号 | 功能模块 | 新增内容 | 提点效果分析 |
|-----|---------|---------|-------------|
| 1 | Target Encoding | K折CTR编码，避免数据泄露 | ⭐⭐⭐⭐⭐ **高效提点**，直接引入标签信息的统计特征 |
| 2 | 缺失值标记 | 9个高缺失列的 `_missing` 二值特征 | ⭐⭐⭐ 帮助模型区分缺失与非缺失样本的行为差异 |
| 3 | 时间特征增强 | `is_weekend`, `time_period` (5个时段) | ⭐⭐⭐ 捕捉周末/时段的消费行为差异 |
| 4 | Count对数变换 | `_count_log` 系列特征 | ⭐⭐ 减少长尾分布影响，提升模型稳定性 |
| 5 | 交叉特征 | 4组交叉：user×cate, city×cate, period×cate, is_same_city | ⭐⭐⭐⭐ 捕捉用户/地域偏好的细粒度信息 |
| 6 | 数值分箱 | `distance_bin`, `price_bin` | ⭐⭐ 将连续变量离散化，便于模型学习非线性关系 |
| 7 | 温度特征 | `temp_comfort`, `temp_range` | ⭐ 外卖场景下天气对消费的影响 |

### 详细说明

#### 1. Target Encoding (CTR编码)
```python
class CTRFeatureEncoder:
    """基于K折的Target Encoding编码器，避免数据泄露"""
    def __init__(self, cols, n_folds=5, smoothing=20):
        # 使用贝叶斯平滑处理低频类别
        smooth_ctr = (sum + smoothing * global_mean) / (count + smoothing)
```
**编码的列**: `dtype`, `cate_1`, `cate_2`, `cate_3`, `cityid`, `weather`, `time_period`, `age`, `level`, `gender`

**效果分析**: Target Encoding 是最有效的特征增强手段之一，通过引入标签统计信息，显著提升模型对类别特征的利用效率。K折交叉避免了训练集上的标签泄露问题。

#### 2. 交叉特征
```python
# 城市匹配
df['is_same_city'] = (df['cityid'] == df['loc_cityid']).astype(np.int8)
# 用户-类别交叉
df['user_cate1'] = df['userid'].astype(str) + '_' + df['cate_1'].astype(str)
# 城市-类别交叉  
df['city_cate1'] = df['cityid'].astype(str) + '_' + df['cate_1'].astype(str)
# 时段-类别交叉
df['period_cate1'] = df['time_period'].astype(str) + '_' + df['cate_1'].astype(str)
```

**效果分析**: 交叉特征能捕捉特征间的交互关系，例如"用户A偏好类别B"的细粒度信息，对于推荐场景尤为重要。

### 模型参数调整

| 参数 | V1 | V2 | 调整原因 |
|-----|----|----|---------|
| `iterations` | 15000 | 10000 | 特征增强后收敛更快 |
| `learning_rate` | 0.0626 | 0.05 | 降低以适应更多特征 |
| `depth` | 10 | 8 | 减少过拟合风险 |
| `l2_leaf_reg` | 3.69 | 5.0 | 增加正则化 |
| `boosting_type` | Ordered | Plain | 加速训练 |
| `one_hot_max_size` | 2 | 10 | 增加低基数类别的one-hot编码 |
| `od_wait` | 500 | 300 | 减少early stopping等待 |
| `min_data_in_leaf` | - | 50 | 新增，防止过拟合 |

---

## V3 升级内容分析（V2 → V3）

**文件**: `train_catboost_v3.py`

### 新增功能一览

| 序号 | 功能模块 | 新增内容 | 提点效果分析 |
|-----|---------|---------|-------------|
| 1 | 扩展交叉特征 | 11组二阶交叉（用户属性×物品属性） | ⭐⭐⭐⭐ 更全面地捕捉用户偏好 |
| 2 | 对数数值特征 | 5个长尾数值的 `_log` 变换 | ⭐⭐⭐ 处理长尾分布，提升模型对异常值的鲁棒性 |
| 3 | 相对统计特征 | 价格/浏览量相对于类别/城市/等级的偏差 | ⭐⭐⭐⭐ 捕捉"相对贵/便宜"的语义信息 |
| 4 | 扩展Target Encoding | 新增 `price_bin`, `distance_bin` 的CTR | ⭐⭐⭐ 更多维度的标签统计 |

### 详细说明

#### 1. 扩展交叉特征
```python
cross_pairs = [
    ('userid', 'cate_1'), ('userid', 'cate_2'),  # 用户×一/二级类别
    ('cityid', 'cate_1'), ('cityid', 'cate_2'),  # 城市×类别
    ('time_period', 'cate_1'), ('time_period', 'dtype'),  # 时段×类别/类型
    ('gender', 'cate_1'), ('age', 'cate_1'), ('level', 'cate_1'),  # 用户属性×类别
    ('gender', 'dtype'), ('age', 'dtype')  # 用户属性×物品类型
]
```

**效果分析**: V3 将交叉特征从 4 组扩展到 11 组，特别增加了用户人口属性（性别、年龄、等级）与物品类别的交叉，这对于刻画"不同人群对不同品类的偏好"至关重要。

#### 2. 对数数值特征
```python
log_cols = ['distance', 'price', 'item_ave_price', 'user_home_dis', 'user_work_dis']
for col in log_cols:
    df[f'{col}_log'] = np.log1p(np.maximum(df[col], 0)).astype(np.float32)
```

**效果分析**: 对数变换能有效压缩长尾分布，使模型更好地学习中间区间的样本，同时减少极端值的影响。

#### 3. 相对统计特征
```python
# 价格相对于类别均值的比值
df['price_relative_to_cate1'] = df['price'] / (cate1_price_mean + 1)
# 价格与类别均值的差值
df['price_diff_cate1'] = df['price'] - cate1_price_mean
# 价格相对于城市均值
df['price_relative_to_city'] = df['price'] / (city_price_mean + 1)
# 浏览量相对于用户等级
df['disp_relative_to_level'] = df['user_displayed_item_num'] / (level_disp_mean + 1)
```

**效果分析**: 相对特征能表达"这个商品在该类别/城市中是贵还是便宜"，比绝对价格更有业务意义。例如，50元在快餐类目是贵的，但在正餐类目是便宜的。

### 模型参数调整

| 参数 | V2 | V3 | 调整原因 |
|-----|----|----|---------|
| `iterations` | 10000 | 12000 | 增加迭代上限 |
| `learning_rate` | 0.05 | 0.04 | 进一步降低以适应更多特征 |
| `l2_leaf_reg` | 5.0 | 7.0 | 增加正则化防止过拟合 |
| `od_wait` | 300 | 400 | 增加等待轮数 |
| `bagging_temperature` | 0.3 | 0.4 | 增加随机性 |
| `min_data_in_leaf` | 50 | 60 | 增加叶节点最小样本 |

---

## V4 升级内容分析（V3 → V4）

**文件**: `train_catboost_v4.py`

### 新增功能一览

| 序号 | 功能模块 | 新增内容 | 提点效果分析 |
|-----|---------|---------|-------------|
| 1 | **移除无效特征** | 删除 sin/cos 周期编码（重要性为0） | ⭐⭐ 减少噪声特征，提升模型效率 |
| 2 | **细化时段分类** | 7级时段（早高峰/上午/午餐/下午/晚高峰/晚间/深夜） | ⭐⭐ 更精细的时间行为刻画 |
| 3 | **工作日/用餐特征** | `is_rush_hour`, `is_meal_time` | ⭐⭐⭐ 外卖场景的核心时间特征 |
| 4 | **用户/物品曝光统计** | impressions, is_new, activity_bin（无泄露） | ⭐⭐⭐⭐ 区分新老用户/热门冷门物品 |
| 5 | **排名特征** | 类别内/城市内的热度排名及百分位 | ⭐⭐⭐⭐ 物品的相对热度信息 |
| 6 | **新增交叉特征** | weekday×time_period, is_weekend×time_period | ⭐⭐ 更细粒度的时间交互 |
| 7 | **价格Z-score** | 价格在类别内的标准化得分 | ⭐⭐⭐ 更标准的相对价格度量 |
| 8 | **距离相对特征** | 距离相对于类别均值 | ⭐⭐ 补充距离维度的相对信息 |
| 9 | **统计聚合特征** | 类别价格/城市距离的均值、标准差、极值 | ⭐⭐⭐ 丰富的上下文统计信息 |
| 10 | **高级CTR编码器** | 支持多列交叉组合的Target Encoding | ⭐⭐⭐⭐⭐ 更强的标签统计能力 |
| 11 | **交叉组合CTR** | 7组组合CTR（city×cate, gender×cate等） | ⭐⭐⭐⭐⭐ 高阶交互的标签统计 |
| 12 | **加权集成** | 根据各折AUC进行softmax加权平均 | ⭐⭐⭐ 提升集成效果 |
| 13 | **检查点续训** | 支持训练中断后恢复 | 工程优化 |

### 详细说明

#### 1. 移除无效特征
```python
# V4: 不再使用sin/cos (在V3中重要性为0)
# 改用更直接的时间段特征
```

**效果分析**: sin/cos 周期编码在树模型中通常效果不佳，因为树模型难以学习三角函数的周期性。删除这些特征可以减少模型噪声。

#### 2. 用户/物品历史行为统计（无泄露版本）
```python
# ⚠️ 修复：只使用曝光次数（count），不使用label相关的统计
user_impressions = train_df.groupby('userid').size()  # 无泄露
df['is_new_user'] = (df['user_impressions'] <= 1).astype(np.int8)
df['user_activity_bin'] = pd.cut(df['user_impressions'], 
                                  bins=[-1, 1, 2, 5, 10, 100],
                                  labels=[0, 1, 2, 3, 4])
```

**效果分析**: 区分新老用户和热门冷门物品是推荐系统的核心需求。新用户/新物品往往需要不同的推荐策略。这里特别注意避免了使用label进行统计导致的数据泄露问题。

#### 3. 排名特征
```python
# 物品在类别内的热度排名
cate_item_rank['cate_item_rank'] = cate_item_rank.groupby('cate_1')['cate_item_count'].rank(
    method='dense', ascending=False
)
# 排名百分位
cate_item_rank['cate_item_rank_pct'] = cate_item_rank.groupby('cate_1')['cate_item_count'].rank(
    method='dense', pct=True
)
```

**效果分析**: 排名特征提供了物品在同类中的相对位置信息，有助于模型理解"这是该类别的热门商品还是冷门商品"。

#### 4. 高级Target Encoding编码器
```python
class AdvancedCTREncoder:
    """支持单列编码和多列交叉编码"""
    def fit_transform_single(self, X, y, cols, prefix=''):
        # 创建组合键
        if len(cols) == 1:
            combined_col = X[cols[0]].astype(str)
        else:
            combined_col = X[cols[0]].astype(str)
            for c in cols[1:]:
                combined_col = combined_col + '_' + X[c].astype(str)
```

#### 5. 交叉组合Target Encoding
```python
cross_ctr_cols = [
    ['cityid', 'cate_1'],    # 城市-类别CTR
    ['cityid', 'dtype'],     # 城市-类型CTR
    ['gender', 'cate_1'],    # 性别-类别CTR
    ['age', 'cate_1'],       # 年龄-类别CTR
    ['level', 'cate_1'],     # 等级-类别CTR
    ['time_period', 'dtype'], # 时段-类型CTR
    ['weekday', 'hour'],     # 星期-小时CTR
]
```

**效果分析**: 交叉组合CTR是V4最重要的提升之一。它能捕捉"北京用户对火锅的偏好"、"女性用户对甜品的偏好"等高阶交互信息，直接编码为可用特征。

#### 6. 加权集成预测
```python
def weighted_ensemble_predict(models, cv_scores, X_test, categorical_features):
    # 使用softmax计算权重
    scores = np.array(cv_scores)
    weights = np.exp((scores - scores.min()) * 100)  # 放大差异
    weights = weights / weights.sum()
    # 加权平均
    ensemble_pred = np.sum(predictions * weights, axis=1)
```

**效果分析**: 加权集成让验证集表现更好的模型在最终预测中占更大权重，相比简单平均能带来微小但稳定的提升。

### 模型参数调整

| 参数 | V3 | V4 | 调整原因 |
|-----|----|----|---------|
| `iterations` | 12000 | 15000 | 进一步增加迭代上限 |
| `learning_rate` | 0.04 | 0.03 | 更低的学习率配合更多特征 |
| `l2_leaf_reg` | 7.0 | 8.0 | 更强的正则化 |
| `od_wait` | 400 | 500 | 更长的early stopping等待 |
| `bagging_temperature` | 0.4 | 0.5 | 更高的随机性 |
| `min_data_in_leaf` | 60 | 80 | 更大的叶节点约束 |
| `grow_policy` | - | SymmetricTree | 新增对称树策略 |

---

## V4 vs V1 全部新增功能总结

从 V1 到 V4 的完整升级路线可以归纳为以下 **8 大类**优化：

### 一、Target Encoding 体系

| 能力 | V1 | V4 |
|-----|----|----|
| 基础CTR编码 | ❌ | ✅ 10+单列CTR |
| 交叉组合CTR | ❌ | ✅ 7组交叉CTR |
| K折防泄露 | ❌ | ✅ 严格K折 |
| 贝叶斯平滑 | ❌ | ✅ smoothing=20 |

**提点贡献**: ⭐⭐⭐⭐⭐ (最大贡献)

### 二、交叉特征工程

| 能力 | V1 | V4 |
|-----|----|----|
| 字符串交叉 | ❌ | ✅ 12+组 |
| 城市匹配 | ❌ | ✅ is_same_city |
| 时间交叉 | ❌ | ✅ weekday×period等 |

**提点贡献**: ⭐⭐⭐⭐

### 三、用户/物品行为特征

| 能力 | V1 | V4 |
|-----|----|----|
| 曝光次数统计 | ❌ | ✅ user/item impressions |
| 新用户/新物品标识 | ❌ | ✅ is_new_user/item |
| 活跃度分箱 | ❌ | ✅ activity_bin |
| 热度排名 | ❌ | ✅ rank/rank_pct |

**提点贡献**: ⭐⭐⭐⭐

### 四、相对/统计特征

| 能力 | V1 | V4 |
|-----|----|----|
| 相对价格 | ❌ | ✅ price_relative_to_* |
| 价格Z-score | ❌ | ✅ price_zscore_cate1 |
| 相对距离 | ❌ | ✅ dist_relative_to_cate1 |
| 统计聚合 | ❌ | ✅ mean/std/min/max |

**提点贡献**: ⭐⭐⭐

### 五、时间特征增强

| 能力 | V1 | V4 |
|-----|----|----|
| sin/cos编码 | ✅ | ❌ (移除) |
| 时段分类 | ❌ | ✅ 7级精细时段 |
| 周末标识 | ❌ | ✅ is_weekend |
| 高峰/用餐时间 | ❌ | ✅ is_rush_hour/meal_time |

**提点贡献**: ⭐⭐⭐

### 六、数值特征处理

| 能力 | V1 | V4 |
|-----|----|----|
| 对数变换 | ❌ | ✅ 5列_log特征 |
| 精细分箱 | ❌ | ✅ 更多区间 |
| 缺失值标记 | ❌ | ✅ 9列_missing |
| Count对数 | ❌ | ✅ _count_log |

**提点贡献**: ⭐⭐

### 七、模型训练优化

| 能力 | V1 | V4 |
|-----|----|----|
| 学习率 | 0.0626 | 0.03 |
| 树深度 | 10 | 8 |
| 正则化 | 3.69 | 8.0 |
| 叶节点约束 | 无 | 80 |
| early stopping | 500 | 500 |
| grow_policy | - | SymmetricTree |

**提点贡献**: ⭐⭐⭐

### 八、集成策略优化

| 能力 | V1 | V4 |
|-----|----|----|
| 集成方式 | 简单平均 | AUC加权平均 |
| 权重计算 | - | Softmax |

**提点贡献**: ⭐⭐

---

## 版本对比总表

| 维度 | V1 | V2 | V3 | V4 |
|-----|----|----|----|----|
| **特征数量** | ~30 | ~60 | ~90 | ~130+ |
| **Target Encoding** | ❌ | 单列 | 扩展单列 | 单列+交叉 |
| **交叉特征** | ❌ | 4组 | 11组 | 12+组 |
| **用户/物品统计** | ❌ | ❌ | ❌ | ✅ |
| **排名特征** | ❌ | ❌ | ❌ | ✅ |
| **相对特征** | ❌ | ❌ | ✅ | ✅增强 |
| **时间特征** | sin/cos | 时段分类 | 同V2 | 精细7级+高峰 |
| **对数变换** | ❌ | count_log | 数值_log | 同V3 |
| **统计聚合** | ❌ | ❌ | ❌ | ✅ |
| **集成策略** | 简单平均 | 简单平均 | 简单平均 | 加权平均 |
| **模型深度** | 10 | 8 | 8 | 8 |
| **正则化** | 3.69 | 5.0 | 7.0 | 8.0 |
| **学习率** | 0.063 | 0.05 | 0.04 | 0.03 |

---

## 总结

从 V1 到 V4 的升级过程中，**最核心的提升来源于**：

1. **Target Encoding 体系的建立和扩展** - 从无到单列再到交叉组合CTR
2. **交叉特征工程** - 捕捉用户×物品、时间×类别等交互关系
3. **用户/物品行为统计** - 区分冷热门物品和新老用户
4. **相对统计特征** - 让模型理解"相对贵/便宜"的业务语义
5. **模型正则化逐步加强** - 随着特征增加防止过拟合

整体设计遵循了 **"特征越丰富，正则化越强"** 的原则，确保模型在利用更多信息的同时保持泛化能力。
