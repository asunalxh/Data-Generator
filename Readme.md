# 简单数据生成器
该项目能够根据有向无环图的依赖结构生成数据集，但仅限于非常简单的场景

## 数据场景

- 数据均为离散数据，由自然数表示，可生成后再映射字符串
- 依赖关系仅支持由多到一的映射关系，例如$(A,B) \rightarrow C$
- 依赖关系的映射仅支持设置最高的概率，例如$(A=0,b=1)$前提下，$C$最可能为$1$，其余可能的$C$值可设置正态分布、均匀分布的常见分布


## 配置文件
配置文件为json格式文件
```json
{
  // 属性值域大小
  "domain": {
    //"属性名": 大小
    "A": 19,
    "C": 2
  },
  // 指定属性的分布情况
  "distribution": {
    "C": {
      //依赖项，可以为空
      "from": [
        "A",
        "B"
      ],
      //生成项最大概率值（特殊值）的概率设置，概率值将从区间均匀随机产生，可为空
      //该值的含义为，对任意A,B属性的组合，对应C中随机某一特殊值的概率，比如(A=0,b=0)中C=0的概率，此特殊值由随机选择产生
      "max_dependency_internal": [
        0.80,
        0.95
      ],
      //生成项非特殊值以外的概率分布，可选择normal,uniform，默认为uniform
      "default": "normal"
    },
    "A": {
      "from": [],
      "max_dependency_internal": [],
      "default": "uniform"
    }
  }
}

```
