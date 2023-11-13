---
sticky: -2
title: python现在的无头浏览器启用
date: 2023-11-13 09:11:53
updated: 2023-11-13 09:11:53
tags:
  - python
  - selenium
comments: true
---
第一步，先安装python

第二部，pip 安装selenium pip install -U selenium 安装升级selenium的需求。

第三步 下载对应浏览器的webdriver

https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/?ch=1

我用的是EDGE，webdriver对应edge的版本。

第四步 测试启动 driver = webdriver.Edge()

driver.implicitly_wait(15)
driver.get("https://www.baidu.com")

能看到浏览器启动打开百度即可