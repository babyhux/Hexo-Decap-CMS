---
title: SO-VITS-SVC详细安装、训练、推理使用步骤
date: 2023-06-01 09:40:02
updated: ""
tags:
  - Ai
comments: true
---
<!--StartFragment-->

由于B站不支持Markdown语言，大家可以去下面网站看 https://github.com/SUC-DriverOld/so-vits-svc-Chinese-Detaild-Documents https://blog.csdn.net/Sucial/article/details/129104472

本帮助文档为项目 [so-vits-svc](https://github.com/innnky/so-vits-svc) 的详细中文安装、调试、推理教程，您也可以直接选择官方[README](https://github.com/innnky/so-vits-svc#readme)文档

撰写：Sucial [点击跳转B站主页](https://space.bilibili.com/445022409)

以下仅展示部分文档内容，原文档很详细！！！

1. 环境依赖

> * \*\*本项目需要的环境：
>
> NVIDIA-CUDA
>
> Python <= 3.10
>
> Pytorch
>
> FFmpeg

Cuda

在cmd控制台里输入`nvidia-smi.exe`以查看显卡驱动版本和对应的cuda版本

前往 [NVIDIA-Developer](https://developer.nvidia.com/) 官网下载与系统**对应**的Cuda版本

安装成功之后在cmd控制台中输入`nvcc -V`, 出现类似以下内容则安装成功：

特别注意！

* 目前为止pytorch最高支持到`cuda11.7`
* 如果您在上述第一步中查看到自己的Cuda版本>11.7，请依然选择11.7进行下载安装（Cuda有版本兼容性）并且安装完成后再次在cmd输入`nvidia-smi.exe`并不会出现cuda版本变化，即任然显示的是>11,7的版本
* \*\*Cuda的卸载方法：\*\*打开控制面板-程序-卸载程序，将带有`NVIDIA CUDA`的程序全部卸载即可（一共5个）

Python

* 前往 [Python官网](https://www.python.org/) 下载Python，版本需要低于3.10（详细安装方法以及添加Path此处省略，网上随便一查都有）
* 安装完成后在cmd控制台中输入`python`出现类似以下内容则安装成功：
* 配置python下载镜像源（有国外网络条件可跳过）

在cmd控制台依次执行

pip config set global.index-url <http://pypi.tuna.tsinghua.edu.cn/simple>

pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

安装依赖库

* 在任意位置新建名为`requirements.txt`的文本文件，输入以下内容保存

```shell

```

* 在该文本文件所处文件夹内右击空白处选择 **在终端中打开** 并执行下面命令以安装库（若出现报错请尝试用`pip install [库名称]`重新单独安装直至成功）

```shell

```

* 接下来我们需要**单独安装**`torch`, `torchaudio`, `torchvision`这三个库，下面提供两种方法

方法1（便捷但不建议，因为我在测试这种方法过程中发现有问题，对后续配置AI有影响

> 直接前往 [Pytorch官网](https://pytorch.org/get-started/locally/) 选择所需版本然后复制Run this Command栏显示的命令至cmd安装（不建议）

方法2（较慢但稳定，建议）

* 前往该地址使用`Ctrl+F`搜索直接下载whl包 [点击前往](https://download.pytorch.org/whl/)

> * 这个项目需要的是
>
> `torch==1.10.0+cu113`
>
> `torchaudio==0.10.0+cu113`
>
> 1.10.0 和0.10.0表示是pytorch版本，cu113表示cuda版本11.3
>
> 以此类推，请选择**适合自己的版本**安装

* **下面我将以`Cuda11.7`版本为例**

***\--示例开始–***

> * 我们需要安装以下三个库
>
> 1. `torch-1.13.0+cu117` 点击下载：[torch-1.13.0+cu117-cp310-cp310-win_amd64.whl](https://download.pytorch.org/whl/cu117/torch-1.13.0%2Bcu117-cp310-cp310-win_amd64.whl)

其中cp310指`python3.10`, `win-amd64`表示windows 64位操作系统

> 2. `torchaudio-0.13.0+cu117`点击下载：[torchaudio-0.13.0+cu117-cp310-cp310-win_amd64.whl](https://download.pytorch.org/whl/cu117/torchaudio-0.13.0%2Bcu117-cp310-cp310-win_amd64.whl)
> 3. `torchvision-0.14.0+cu117`点击下载：[torchvision-0.14.0+cu117-cp310-cp310-win_amd64.whl](https://download.pytorch.org/whl/cu117/torchvision-0.14.0%2Bcu117-cp310-cp310-win_amd64.whl)

* 下载完成后进入进入下载的whl文件的目录，在所处文件夹内右击空白处选择 **在终端中打开** 并执行下面命令以安装库

```shell

```

* 务必在出现`Successfully installed ...`之后再执行下一条命令，第一个torch包安装时间较长

***\--示例结束–***

安装完`torch`, `torchaudio`, `torchvision`这三个库之后，在cmd控制台运用以下命令检测cuda与torch版本是否匹配

```shell

```

* 最后一行出现`True`则成功，出现`False`则失败，需要重新安装

FFmpeg

* 前往 [FFmpeg官网](https://ffmpeg.org/) 下载。解压至任意位置并在高级系统设置-环境变量中添加Path定位至`.\ffmpeg\bin`（详细安装方法以及添加Path此处省略，网上随便一查都有）
* 安装完成后在cmd控制台中输入`ffmpeg -version`出现类似以下内容则安装成功

2. 预训练AI

下载项目源码

* 前往 [so-vits-svc](https://github.com/innnky/so-vits-svc) 选择`32k`分支（本教程针对`32k`）下载源代码。安装了git的可直接git以下地址
* 解压到任意文件夹

下载预训练模型

* 这部分官方文档写得很详细，我这边直接引用

> **hubert**
>
> <https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt>
>
> **G与D预训练模型**
>
> <https://huggingface.co/innnky/sovits_pretrained/resolve/main/G_0.pth>
>
> <https://huggingface.co/innnky/sovits_pretrained/resolve/main/D_0.pth>

三个训练底模下载不了的进这个链接：https://pan.baidu.com/s/1uw6W3gOBvMbVey1qt_AzhA?pwd=80eo提取码：80eo

* `hubert-soft-0d54a1f4.pt`放入`.\hubert`文件夹
* `D_0.pth和G_0.pth`文件放入`.\logs\32k`文件夹

准备训练样本

> 准备的训练数据，建议60-100条语音(**格式务必为wav，不同的说话人建立不同的文件夹**)，每条语音控制在**4-8秒！**（确保语音不要有噪音或尽量降低噪音，一个文件夹内语音必须是一个人说的），可以训练出效果不错的模型

* 将语音连带文件夹（有多个人就多个文件夹）一起放入`.\dataset_raw`文件夹里，文件结构类似如下：

```shell

```

* 此外还需要在`.\dataset_raw`文件夹内新建并编辑`config.json`，代码如下：

```shell

```

样本预处理

下面的所有步骤若出现报错请多次尝试，若一直报错就是第一部分环境依赖没有装到位，可以根据报错内容重新安装对应的库。（一般如果正确安装了的话出现报错请多次尝试或者关机重启，肯定可以解决报错的。）

1. 重采样

* 在`so-vits-svc`文件夹内运行终端，直接执行：

    python resample.py

**注意：如果遇到如下报错：**

E:\vs\so-vits-svc-32k\resample.py:17: FutureWarning: Pass sr=None as keyword args. From version 0.10 passing these as positional arguments will result in an error

  wav, sr = librosa.load(wav_path, None)

E:\vs\so-vits-svc-32k\resample.py:17: FutureWarning: Pass sr=None as keyword args. From version 0.10 passing these as positional arguments will result in an error

  wav, sr = librosa.load(wav_path, None)

请打开`resample.py`，修改第`17`行内容

# 第17行修改前如下

wav, sr = librosa.load(wav_path, None)

# 第17行修改后如下

wav, sr = librosa.load(wav_path, sr = None)

保存，重新执行`python resample.py`命令

* 成功运行后，在`.\dataset\32k`文件夹中会有说话人的wav语音，之后`dataset_raw`文件夹就可以删除了

2. 自动划分训练集，验证集，测试集，自动生成配置文件

* 在`so-vits-svc`文件夹内运行终端，直接执行：

    python preprocess_flist_config.py

3. 生成hubert和f0

* 在`so-vits-svc`文件夹内运行终端，直接执行：

    python preprocess_hubert_f0.py

4. 修改配置文件和部分源代码

* 打开上面第二步过程中生成的配置文件`.\configs\config.json`修改第`13`行代码`"batch_size"`的数值。这边解释一下`"batch_size": 12,`数值12要根据自己电脑的显存（任务管理器-GPU-**专用**GPU内存）来调整

> * **修改建议**
>
> 6G显存 建议修改成2或3
>
> 8G显存 建议修改成4
>
> “batch_size"参数调小可以解决显存不够的问题

* 修改`train.py`

```shell

```

3. 开始训练

* 在`so-vits-svc`文件夹内运行终端，直接执行下面命令开始训练

**注意：开始训练前建议重启一下电脑清理内存和显存，并且关闭后台游戏，动态壁纸等等软件，最好只留一个cmd窗口**

    python train.py -c configs/config.json -m 32k

* 出现以下报错就是显存不够了

```shell

```

> **以下的解释我引用了B站up主inifnite_loop的解释，[相关视频](https://www.bilibili.com/video/BV1Bd4y1W7BN) [相关专栏](https://www.bilibili.com/read/cv21425662)**
>
> * 需要关注两个参数：Epoch和global_step
>
> Epoch表示迭代批次，每一批次可以看作一个迭代分组
>
> Global_step表示总体迭代次数
>
> * 两者的关系是global_step = 最多语音说话人的语音数 /  batch_size  * epoch
>
> batch_size是配置文件中的参数
>
> * **示例1:** 每一次迭代输出内 `====> Epoch: 74` 表示第74迭代批次完成
> * **示例2:** `Global_step` 每200次输出一次 （配置文件中的参数`log_interval`）
> * **示例3:** `Global_step` 每1000次输出一次（配置文件中的参数`eval_interval`），会保存模型到新的文件

一般情况下训练10000次（大约2小时）就能得到一个不错的声音模型了

保存的训练模型

> 以上，我们谈论到了每1000次迭代才会保存一次模型样本，那么，这些样本保存在哪里呢？如何处理这些样本呢？下面我将详细讲述。

* 训练模型保存位置：`.\logs\32k`
* 训练一定时间后打开这个路径，你会发现有很多文件：

```shell

```

推理生成

* 修改完成后保存代码，在`so-vits-svc`文件夹内运行终端，执行下面命令开始推理生成

    python .\inference_main.py

* 待黑窗口自动关闭后，推理生成完成。生成的音频文件在`.\results`文件夹下
* 如果听上去效果不好，就多训练模型，10000次不够就训练20000次

后期处理

* 将生成的干音和歌曲伴奏（也可以通过Ultimate Vocal Remover提取）导入音频处理软件&宿主软件（如Au，Studio One等）进行混音和母带处理，最终得到成品。

5. 感谢名单

> * **以下是对本文档的撰写有帮助的感谢名单：**
>
> so-vits-svc [官方源代码和帮助文档](https://github.com/innnky/so-vits-svc)
>
> B站up主inifnite_loop [相关视频](https://www.bilibili.com/video/BV1Bd4y1W7BN) [相关专栏](https://www.bilibili.com/read/cv21425662)
>
> 所有提供训练音频样本的人员 作者：Sucial丶 <https://www.bilibili.com/read/cv21907650> 出处：bilibili

<!--EndFragment-->