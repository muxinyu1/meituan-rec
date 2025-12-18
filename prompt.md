编写代码训练rankmixer，训练集在train.csv

需要删除的特征：

- timestamp
- geohash
- work_geohash
- userid

数值特征：
- distance
- item_ave_price
- price
- user_home_dis
- user_work_dis
- temp
- temp_low
- temp_high
- user_displayed_item_num
- online_days

剩下的就是类别特征。

itemid数量有十万，实际上是很多的，为了减小参数量，做一个简单的特征工程：把出现次数小于某个值的itemid统一干成<UNK>，使得总的不同itemid不会太高。

注意事项：

- 缺失值填充，对于确实率小于10%的特征，使用中位数，否则新建一列特征xxx_is_missing
- 需要划分验证集，划分的方式是保证训练集和验证集的weekday和hour分布大体一致
- 需要对数值特征分桶
- 特征需要压缩，比如1, 10, 100压缩为0, 1, 2

先不做特征工程。
