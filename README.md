# 仿真平台


## 环境配置
支持和测试的平台包括：Win10 和 Ubuntu 24.04。
1. 下载并安装 Python 3.7、Carla 0.9.15、latex 2023、Texstudio 4.6.4、Git 2.42.0。
2. 使用`git clone https://github.com/OpenHUTB/sim.git` 进行仓库的递归克隆。

## 注意事项
代码目录结构
```dtd
{MODULE_NAME}/
    undergraduate/
        fig/
            carla.pptx
            generate_img.py
        hutbthesis_main.tex
    README.md
test/
sim.config
README.md
```
1. 每个模块的使用和说明文档以`*.md`格式更新到 [文档仓库](https://github.com/OpenHUTB/carla_doc) `docs/{MODULE_NAME}`目录下。
2. 仓库中只保留*.md、*.py、*.cpp、*.conf、*.sh、*.bat等文件，最多保留一个演示数据，其他数据都保留到百度网盘并提供链接和地址到README.md文件中。
3. 每次提交统一用Pull Request的方式进行，至少需要一个人的进行Code Review和测试才能合并到主分支。
4. 每个项目主要代码都放在一个目录下，保证每次更新时整个项目都能运行（写功能代码前先写测试代码和样例），每个模块只有一个Python脚本作为入口程序。
5. 作图统一使用 PPT 或 TikZ 绘图，放置在目录`{MODULE_NAME}/undergraduate/fig/`目录下，使用`generate_img.py`脚本生成相应的`*.eps`格式作为统一的图片格式（不需要上传）。
6. 代码中不要使用绝对路径，统一使用相对路径。
7. 所有模块的配置统一放到`sim.config`文件中。


## 参考链接

- [中文文档](https://openhutb.github.io/carla_doc/)
- [毕业论文模板](https://github.com/OpenHUTB/undergraduate)
