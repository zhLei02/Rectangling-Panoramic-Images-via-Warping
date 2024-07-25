#ifndef common_h
#define common_h

// 无穷大
#define INF 1e8
// Π
#define PI 3.14159265358979323846
// mask中有无像素标记值
#define missingpixel 0
#define haspixel 255
// mesh边颜色
#define mesh_edge_color Scalar(0,255,0) // 颜色：绿色

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Eigen是一个高效的C++线性代数库，广泛用于数值计算、科学计算和计算机图形学
#include <Eigen/Sparse> // 稀疏矩阵运算
#include <Eigen/Dense> // 稠密矩阵运算

#include <GL/glew.h>
#include <GLFW/glfw3.h>

using namespace std;
using namespace cv;
using namespace Eigen;

// 使用泛洪填充将mask内部可能的坑洞填充
Mat fillHole(const Mat& mask);

// 获取图像的mask
Mat getMask(const Mat& img);

// 坐标类
class Coordinate
{
public:
    int row;
    int col;

    // 构造函数
    Coordinate()
    {
        row=0;
        col=0;
    }

    Coordinate(int row, int col)
    {
        this->row = row;
        this->col = col;
    }

    // 重载运算符
    bool operator==(const Coordinate& c)
    {
        return (this->row == c.row) && (this->col == c.col);
    }

    Coordinate operator+=(const Coordinate& c)
    {
        this->row += c.row;
        this->col += c.col;
        return *this;
    }

    Coordinate operator-(const Coordinate& c) const
    {
        return Coordinate(this->row - c.row, this->col - c.col);
    }
};

// 坐标类(double)，用于处理网格顶点
class CoordinateDouble
{
public:
    double row;
    double col;

    // 构造函数
    CoordinateDouble()
    {
        row=0;
        col=0;
    }

    CoordinateDouble(double row, double col)
    {
        this->row = row;
        this->col = col;
    }

    // 重载运算符
    bool operator==(const CoordinateDouble& c)
    {
        return (this->row == c.row) && (this->col == c.col);
    }

    CoordinateDouble operator+=(const CoordinateDouble& c)
    {
        this->row += c.row;
        this->col += c.col;
        return *this;
    }

    CoordinateDouble operator-(const CoordinateDouble& c) const
    {
        CoordinateDouble res;
        res.row = this->row - c.row;
        res.col = this->col - c.col;
        return res;
    }

    CoordinateDouble operator+(const CoordinateDouble& c) const
    {
        CoordinateDouble res;
        res.row = this->row + c.row;
        res.col = this->col + c.col;
        return res;
    }
};

// 运行前配置类
class Config
{
public:
    // 图片参数
    int rows;
    int cols;
    // 网格顶点数
    int mesh_rows;
    int mesh_cols;
    // 网格边数
    int mesh_quad_rows;
    int mesh_quad_cols;
    // 网格边长
    double row_len; // 网格的高
    double col_len; // 网格的宽
    // 每个方格内的线段数目
    int lines_per_quad;
    // 迭代次数
    int times;

    Config(){
        lines_per_quad = 25;
        times = 10;
    };

    Config(int rows,int cols,int mesh_rows=20,int mesh_cols=20,int lines_per_quad=25,int times=10)
    {
        this->rows = rows;
        this->cols = cols;
        this->mesh_rows = mesh_rows;
        this->mesh_cols = mesh_cols;
        this->mesh_quad_rows = mesh_rows - 1;
        this->mesh_quad_cols = mesh_cols - 1;
        this->row_len = (double)(rows-1) / mesh_quad_rows;
        this->col_len = (double)(cols-1) / mesh_quad_cols;
        this->lines_per_quad = lines_per_quad;
        this->times = times;
    }
};

// 画直线
void drawLine(Mat& img, const CoordinateDouble& p1, const CoordinateDouble& p2, Scalar color = mesh_edge_color);

// 画方格
Mat drawMesh(Mat img, vector<vector<CoordinateDouble>>& mesh, Config config);

/**
 * 以下为GlobalWraping添加
*/
// 直线类
class LineD
{
public:
    CoordinateDouble p1;
    CoordinateDouble p2;

    // 构造函数
    LineD()
    {
        p1 = CoordinateDouble(0,0);
        p2 = CoordinateDouble(0,0);
    }

    LineD(CoordinateDouble p1, CoordinateDouble p2)
    {
        this->p1 = p1;
        this->p2 = p2;
    }

};

// 双线性插值权重
struct BilinearWeights
{
    double s;
    double t;

    BilinearWeights()
    {
        s = 0;
        t = 0;
    }

    BilinearWeights(double s, double t)
    {
        this->s = s;
        this->t = t;
    }
};

/**
 * 按行堆叠矩阵
*/
Eigen::SparseMatrix<double,Eigen::RowMajor> row_stack(const Eigen::SparseMatrix<double,Eigen::RowMajor>& A,
                                                     const Eigen::SparseMatrix<double,Eigen::RowMajor>& B);

/**
 * 将vec转为mesh
*/
vector<vector<CoordinateDouble>> vec2mesh(const VectorXd& vec, const Config& config);

/**
 * 通过mesh获得Vq
*/
VectorXd get_Vq(const vector<vector<CoordinateDouble>>& mesh, int i, int j);

/**
 * 扩大mesh s倍
*/
void enlarge_mesh(vector<vector<CoordinateDouble>>& mesh, double s, Config& config);

/**
 * 双线性插值获取最终形变后的图像
*/
Vec3f bilinear_interpolation(const Mat& img, double x, double y);

#endif