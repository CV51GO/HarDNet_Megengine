## 介绍

本项目是论文《HarDNet: A Low Memory Traffic Network》的Megengine实现。该论文的官方实现地址：https://github.com/PingoLH/Pytorch-HarDNet


## 环境安装

依赖于CUDA10

```
conda create -n HarDNet python=3.7
pip install -r requirements.txt
```

## 使用方法

安装完环境后，直接运行`python compare.py`。

`compare.py`文件对官方实现和Megengine实现的推理结果进行了对比。

运行`compare.py`时，会读取`./data`中存放的图片进行推理。`compare.py`中实现了Megengine框架和官方使用的Pytorch框架的推理，并判断两者推理结果的一致性。


## 模型加载示例

在model.py中，定义了```get_megengine_hardnet_model```方法，该方法能够利用hub加载模型。
```python
@hub.pretrained(
"https://studio.brainpp.com/api/v1/activities/3/missions/86/files/f96cbc0e-43c5-455f-ab5f-8381269a8e2d"
)
def get_megengine_hardnet_model():
    model_megengine = HarDNet()
    return model_megengine
```

在使用模型时，使用如下代码即可加载权重：
```python
from model import get_megengine_hardnet_model
model_megengine = get_megengine_hardnet_model(pretrained=True)
```
