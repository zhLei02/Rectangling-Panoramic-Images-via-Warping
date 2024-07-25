#ifndef GlobalWraping_h
#define GlobalWraping_h

#include <common.h>
#include <lsd.h>

using namespace std;
using namespace cv;
using namespace Eigen;

class GlobalWraping {
private:
    Config config; // 网格参数配置
    Mat img; // 输入图像
    Mat mask; // mask

public:
    GlobalWraping(){}

    // 构造函数
    GlobalWraping(Mat& img, Mat& mask, Config& config);

    // 论文中的Shape Preservation矩阵
	Eigen::SparseMatrix<double,Eigen::RowMajor> get_shape_mat(const vector<vector<CoordinateDouble>>& mesh);

    // 构造Q矩阵 用于计算Vq = Q*V
    Eigen::SparseMatrix<double,Eigen::RowMajor> get_Q_mat(const vector<vector<CoordinateDouble>>& mesh);

    // 论文中的边界约束
    pair<Eigen::SparseMatrix<double,Eigen::RowMajor>,Eigen::VectorXd> get_boundary_mat(const vector<vector<CoordinateDouble>>& mesh);

    // 感觉会是最难的部分，前面都够折磨我了......
    // 为保证线段尽可能笔直且不破坏原本的平行性，进行一下线段约束

    // 判断线段是否在图像内部
    bool line_in_mask(const LineD& line);

    // 调用lsd算法进行线段检测
    vector<LineD> lsd_detect();

    /*------------------------------用于将线段切割到各网格时的工具函数------------------------------*/
    // 检测某点是否在quad中
    bool is_in_quad(const CoordinateDouble& p, const CoordinateDouble& lefttp, const CoordinateDouble& righttp,
                    const CoordinateDouble& leftbt, const CoordinateDouble& rightbt);

    // 找到线段与线段的交点，用于后续判断线段与quad的交点
    bool get_intersection(const LineD& l1, const LineD& l2, CoordinateDouble& intersection);

    // 找到线段与quad的交点，用于线段切割
    vector<CoordinateDouble> get_intersections(const LineD& line, const CoordinateDouble& lefttp,
                    const CoordinateDouble& righttp, const CoordinateDouble& leftbt, const CoordinateDouble& rightbt);
    
    // 将切割后得到的线段三维数组展平
    vector<LineD> flatten_lines(const vector<vector<vector<LineD>>>& segline);
    /*--------------------------------------------END--------------------------------------------*/
    
    // 将线段切割到各网格中
    vector<vector<vector<LineD>>> segline_inquad(const vector<LineD>& lines, const vector<vector<CoordinateDouble>>& mesh);

    // 初始化线段分割 参数介绍见函数实现
    vector<vector<vector<LineD>>> init_line_seg(const vector<vector<CoordinateDouble>>& mesh, vector<LineD>& lineseg_flatten,
        vector<pair<int,double>>& id_theta, vector<double>& rotate_theta);

    /*-------------------------------用于计算线段约束矩阵的工具函数-------------------------------*/
    // 计算双线性插值权重
    BilinearWeights get_bilinear_weights(const CoordinateDouble& p, const CoordinateDouble& p1,
        const CoordinateDouble& p2, const CoordinateDouble& p3, const CoordinateDouble& p4);

    // 将权重转化为对应的矩阵
    MatrixXd bilinearW2Mat(const BilinearWeights&);
    /*--------------------------------------------END--------------------------------------------*/

    // 线段约束矩阵
    Eigen::SparseMatrix<double,Eigen::RowMajor> get_line_mat(const vector<vector<CoordinateDouble>>& mesh,
        const vector<vector<vector<LineD>>>& line_seg_mesh, const vector<double>& rotate_theta,
        vector<pair<MatrixXd, MatrixXd>>& BilinearMat_Vec, int& n_lines);

};


#endif