# comfyui-ez-llm

在`models/llm_adapters`文件夹下创建文件夹，例如我这边的`gemma-3-1b-it`
<img width="488" height="147" alt="image" src="https://github.com/user-attachments/assets/f70015a4-5922-4d2a-9af0-a109f2ef21a4" />

这边创建完以后，去[huggingface](https://huggingface.co/google/gemma-3-1b-it/tree/main)或者[modelscope](https://www.modelscope.cn/models/fireicewolf/google-gemma-3-1b-it/files)把所有文件下载到你刚刚创的文件夹(比如`gemma-3-1b-it`这个文件夹)

目前不兼容`cuda-malloc`,所以启动参数里加个`--disable-cuda-malloc`,例如`/root/miniconda3/bin/python main.py --port 6006 --disable-cuda-malloc`这样来启动你的comfyui

启动以后这么用：
<img width="1624" height="1074" alt="image" src="https://github.com/user-attachments/assets/83797633-e5d0-4cf7-94e0-749316f6ab9e" />
这边的model_mode三个如果是gemma3系列直接选auto，兼容了gemma3全系列，别的模型(比如gemma2,llama这种就选text，兼容了所有的纯文本模型，多模态目前只兼容gemma3)
<img width="644" height="130" alt="image" src="https://github.com/user-attachments/assets/2d8acb38-fdc8-4993-8a1d-47a3efd8f839" />

感觉没了,有话请给我提issue(
