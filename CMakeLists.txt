# 最低版本号
cmake_minimum_required (VERSION 3.10)

# 项目名称
project(RectanglingImages)

# 设置OpenGL策略
cmake_policy(SET CMP0072 NEW)

set(OpenCV /home/yeah/opencv/build)
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
# 引入OpenGL
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

# 引入头文件
include_directories(${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${OPENGL_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS})

# 整合源文件
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST)

# 生成静态库或动态库
add_library(SRC ${SRC_LIST})
# add_library(COMMON ./src/common.cpp)
# add_library(GrabCutsrc ./src/GrabCut.cpp)
# add_library(maxflow ${mincut_list})

# 参数介绍
# add_library(lib_name STATIC/SHARED src) 
# 函数作用：生成库。
# 参数lib_name：是要生成的库名称，
# 参数STATIC/SHARED：指定生成静态库或动态库，
# 参数src：指明库的生成所需要的源文件

# 设置生成库文件放在哪里
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)  # ${PROJECT_SOURCE_DIR} 即CMakeLists.txt的目录

# 可通过 link_directories(./lib) 链接对应文件夹下的库文件
# link_directories(${PROJECT_SOURCE_DIR}/lib)
# 可通过 find_library(FUNC_LIB Hello ${PROJECT_SOURCE_DIR}/lib) 查找库文件 默认查找动态库

# 参数介绍
# find_library(var lib_name lib_path1 lib_path2)
# 函数作用：查找库，并把库的绝对路径和名称存储到第一个参数里
# 参数var：用于存储查找到的库

# 设置可执行文件存放位置
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 生成可执行文件
add_executable(main ${PROJECT_SOURCE_DIR}/main.cpp)
# add_executable(test ${PROJECT_SOURCE_DIR}/test.cpp)
# add_executable(GMM_EM ./tmp/GMM_EM.cpp)
# add_executable(GrabCutEx ./tmp/GrabCutEx.cpp)

# 把库链接到可执行文件中(与26 27 行对应)
# target_link_directories(hello ${FUNC_LIB})

# target_link_libraries(GrabCutsrc GMM)
target_link_libraries(main SRC ${OpenCV_LIBS} Eigen3::Eigen ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} glfw)
# target_link_libraries(test SRC ${OpenCV_LIBS} Eigen3::Eigen ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} glfw)