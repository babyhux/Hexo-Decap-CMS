---
title: Python安装crypto出现的问题
date: 2023-10-07 13:59:00
updated: 2023-10-07 13:59:03
tags:
  - 存档
keywords:
  - python
  - crypto
  - Crypto
comments: true
---
<!--StartFragment-->

# 安装配置Crypto不飘红 from Crypto.PublicKey import RSA ModuleNotFoundError: 报错【Python】

其实问题很简单

你找到你的python库路径下面的“crypto”改为“Crypto”

类似C:\Users\xxx\AppData\Roaming\Python Python38\site-packages

但是 我自己win10 是

C:\Users\Administrator\AppData\Local\Programs\Python\Python310\Lib\site-packages

这样的 改了也有效



<!--EndFragment-->