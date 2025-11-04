# 训练步骤

## 下载处理过后的数据集

```bash
wget -O data.zip https://cloud.tsinghua.edu.cn/f/dbf2a7e8f2674ed19870/?dl=1
unzip data.zip
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 训练

### 训练注意力模型（我们自己的模型）

```bash
python train.py
```

### 训练所有模型（投票）


```bash
python train_all.py
```
