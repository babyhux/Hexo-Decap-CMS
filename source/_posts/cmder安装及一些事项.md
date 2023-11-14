---
title: CMDER安装及一些事项
date: 2023-11-14 12:20:31
updated: 2023-11-14 12:20:31
tags:
  - CMDER
comments: true
---
1. 首先下载地址（官网 https://cmder.app/) 可能需要番茄。
2. 下载版本的话，用full版本比较好，带齐全文件。有GIT。
3. 下载后解压到目录下 既可以使用。
4. 配置环境变量：windows系统高级属性里面添加一个CMDER_HOME的系统变量，变量值就是解压目录<!--StartFragment-->

   ![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pbWFnZXMyMDE4LmNuYmxvZ3MuY29tL2Jsb2cvNzYzOTQ1LzIwMTgwNC83NjM5NDUtMjAxODA0MDQxNTM1MTI0MzItMjU1NzI1NDA1LnBuZw?x-oss-process=image/format,png)

   <!--EndFragment-->
5. path添加，path里面添加一个%CMDER_HOME%的环境变量<!--StartFragment-->

   ![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pbWFnZXMyMDE4LmNuYmxvZ3MuY29tL2Jsb2cvNzYzOTQ1LzIwMTgwNC83NjM5NDUtMjAxODA0MDQxNTM1MzIyNjAtNjQxMTIxOTQ5LnBuZw?x-oss-process=image/format,png)

   <!--EndFragment-->
6. windows右键系统菜单添加。在配置完上面的步骤后，用管理员权限打开cmd，然后输入cmder.exr /REGISTER ALL即可。

   以上大概就是这么多了