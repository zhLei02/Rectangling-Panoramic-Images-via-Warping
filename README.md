## 项目目录
+ bin/              生成的可执行文件
+ build/            构建文件
+ include/          头文件
+ lib/              生成的库文件
+ src/              源文件
+ CMakeLists.txt    构建文件
+ main.cpp          主函数
+ README.md         说明
+ test.cpp          测试文件: coding阶段用于单独检测某模块可用性
    
## 快速开始
+ 安装cmake、gcc、g++、make等工具
+ 安装OpenCV库
+ 安装Eigen库
+ 安装OpenGL库
+ 以上列出主要用到的库，详见CMakeLists.txt
+ 创建build目录，cd build
+ cmake ..
+ make
+ 切换到bin目录，cd ../bin
+ ./main [img_path] 运行main程序