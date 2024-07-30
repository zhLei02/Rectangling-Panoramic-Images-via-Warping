#ifndef imgRecting_h
#define imgRecting_h

#include <GlobalWraping.h>
#include <SeamCarving.h>

using namespace cv;
using namespace std;
using namespace Eigen;

class imgRecting
{
private:
    Mat img;
    vector<vector<CoordinateDouble>> mesh;
    vector<vector<CoordinateDouble>> out_mesh;
    Config config;
    // 以下为glfw相关，用于获得变形后的图像
    GLFWwindow* window;
    GLuint textureID; // 纹理ID
public:
    void init_gl(string img_path); // 初始化glfw，用于纹理贴图完成图像变形
    void getTexture(); // 通过Mat img 获取纹理
    void render(); // 渲染
    void display(); // 显示
    void save_img(string path = "./output"); // 保存图片

    
    void recting();

    imgRecting(Config _config):config(_config){}
    ~imgRecting()
    {
        if(textureID)
        {
            glDeleteTextures(1, &textureID);
        }
        glfwDestroyWindow(window);
        glfwTerminate();
    }
};


#endif