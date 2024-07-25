#ifndef SeamCarving_h
#define SeamCarving_h

#include <common.h>

using namespace std;
using namespace cv;

enum class Border
{
    TOP = 0,        // 上
    RIGHT = 1,      // 右
    BOTTOM = 2,     // 下
    LEFT = 3        // 左
};

enum class SeamDirection
{
    VERTICAL = 0,   // 垂直
    HORISENTAL = 1  // 水平
};

class SeamCarving
{
private:
    bool is_transparent(const Mat& mask, int row,int col); // 判断mask的[row][col]处是否为空
    void init_displacement(vector<vector<Coordinate>>& displacement, int rows, int cols); // 初始化偏移量
    // 计算能量 可使用Sobel进行边缘检测（即识别图像中物体的边界或轮廓，图像的梯度信息）
    Mat cal_energy(const Mat& img);
public:
    // 找到最长的边界段，返回起始坐标和终点坐标
    pair<int,int> choose_longestborder(const Mat& img, const Mat& mask, Border& direction);

    // 显示找到的最长边界段
    void show_longestborder(const Mat& img, pair<int,int> begin_end, Border direction);
    
    // 根据border方向确定seamdirection
    SeamDirection get_seamdirection(Border direction);
    
    // 从子图中找到能量最小的seam
    int* get_minimum_seam(Mat& img,Mat& mask, SeamDirection seamdirection, pair<int,int> begin_end);
    
    // 插入seam
    Mat insert_seam(Mat& img, Mat& mask, SeamDirection seamdirection, int* seam, pair<int,int> begin_end, bool shifttoright);

    // 通过不断插入seam以将图像填充为矩形并返回对应的位移矩阵
    vector<vector<Coordinate>> get_displacements(Mat img, Mat& mask);

    // 根据位移矩阵获取矩形图像
    Mat get_wrapped_img(const Mat& img, const vector<vector<Coordinate>>& displacements);

    // 放置网格
    vector<vector<CoordinateDouble>> place_mesh(const Mat& img, const Config& config);

    // 根据位移修改网格
    void wrap_mesh_back(vector<vector<CoordinateDouble>>& mesh, const vector<vector<Coordinate>>& displacements, const Config& config);
};


#endif