## 下载数据

```sh
wget https://cloud.tsinghua.edu.cn/f/a395289a14c34a37af9d/?dl=1 -O data.zip
unzip data.zip
```

## 训练CatBoost

目前最优模型是使用train_catboost.py训练出来的：

```sh
python train_catboost.py
```

生成`submission.csv`

## 优化方向

- 进一步调参
- 做一些特征工程
- 集成学习

## 参考

`只使用数值特征训练lgbm.md`效果不错，这个md里面是一个prompt，给GPT可以生成训练lightgbm的代码，这种特征工程训练出的lightgbm效果仍然不如直接训练catboost，理论上可以对lgbm调参 + catboost集成，目前只试了无调参的lgbm + catboost集成。

> 深度模型在此次比赛中表现不佳，容易过拟合，深度模型的尝试过程在main分支，也可以参考