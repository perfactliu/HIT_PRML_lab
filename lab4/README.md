实验4涉及使用clip作为编码器，因此需要先配置clip模型。  
点击进入[clip 项目工程repository](https://github.com/openai/CLIP),将其中的全部内容迁移至CLIP-CoOP以及CLIP-zero-shot文件夹中。两个文件夹的最终结构应分别为：
```txt
CLIP-CoOP/clip||data||notebooks||tests||hubconf.py||requirements.txt||setup.py||CoOp.py||shot_plot.py
```
以及
```txt
CLIP-zero-shot/clip||data||notebooks||tests||hubconf.py||requirements.txt||setup.py||zero_shot.py
```
由于使用的数据集为caltech101，因此将两个文件夹中的data文件夹替换为lab3中的data文件夹。
