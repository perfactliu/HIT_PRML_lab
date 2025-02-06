## lab6实验文档说明
### 文件说明
- fine_tuned_lora_unet: &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; lora微调模型（运行lora.py获得）
- fine_tuned_dreambooth_unet: &ensp;&ensp;&ensp;dreambooth微调模型（运行dreambooth.py获得）
- lora.py: &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;lora微调代码
- dreambooth.py: &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;dreambooth微调代码
- create_dreambooth_dataset.py: &ensp;生成dreambooth数据集代码
- model_test.py: &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;模型测试代码
### 预训练模型以及lora数据集下载
预训练模型下载（约30G）
```python
from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/stable-diffusion-v1-5')
```
lora数据集下载
```python
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('buptwq/lora-stable-diffusion-finetune',subset_name='default', split='train')
```