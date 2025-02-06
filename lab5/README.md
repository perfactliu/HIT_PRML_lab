实验5涉及使用clip作为编码器，因此需要先配置clip模型。  
点击进入[clip 项目工程repository](https://github.com/openai/CLIP),将其中的全部内容封装至CLIP文件夹中，并置于lab5文件夹下：
```txt
lab5/CLIP/clip||notebooks||tests||hubconf.py||requirements.txt||setup.py...
```
采用stylegan作为解码器，因此需配置stylegan模型。  
点击进入[stylegan 项目工程repository](https://github.com/NVlabs/stylegan3),将其中的全部内容封装至stylegan3-main文件夹中，并置于lab5文件夹下：
```txt
lab5/stylegan3-main/dnnlib||docs||gui_utils||metrics...
```
接下来根据stylegan工程文档下载stylegan预训练模型，将其置于my_face文件夹中：
```txt
lab5/my_face/stylegan3-t-ffhq-1024x1024.pkl
```
在my_face文件夹中，首先运行my_face.py获得预训练的feature_to_latent_model.pth，接下来运行test.py修改风格空间。