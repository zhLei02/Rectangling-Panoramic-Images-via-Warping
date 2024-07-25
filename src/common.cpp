#include <common.h>


// #define DEBUG

/**
 * 使用泛洪填充将mask内部可能的坑洞填充
 * 预处理中会导致图像内部可能存在坑洞，此处即为填补坑洞
 * 例如：mask为（255表示其为图像，0表示丢失的像素）
 * [[0,  0,  0,  0, 0],
 *  [0,255,255,255, 0],
 *  [0,255,  0,255, 0],
 *  [0,255,255,255, 0],
 *  [0,  0,  0,  0, 0]]
 * 
 * floodfill填充后
 * [[255,255,255,255,255],
 *  [255,255,255,255,255],
 *  [255,255,  0,255,255],
 *  [255,255,255,255,255],
 *  [255,255,255,255,255]]
 * 
 * 取反再或即得
 * [[0,  0,  0,  0, 0],
 *  [0,255,255,255, 0],
 *  [0,255,255,255, 0],
 *  [0,255,255,255, 0],
 *  [0,  0,  0,  0, 0]]
*/
Mat fillHole(const Mat& mask)
{
    if(mask.type() != CV_8UC1) // 确保mask为单通道的灰度图
        throw std::runtime_error("fillHole: mask.type() != CV_8UC1");
    
    Mat tmp = Mat::zeros(mask.rows + 2, mask.cols + 2, mask.type());
    mask.copyTo(tmp(Rect(1, 1, mask.cols, mask.rows)));

    floodFill(tmp, Point(0, 0), Scalar(255)); // 使用泛洪填充将丢失像素填充为255
    Mat res = tmp(Rect(1, 1, mask.cols, mask.rows));
    return (~res) | mask; // 取反后再与原mask进行或运算即可填补坑洞
}

/**
 * 获取图像mask，标注有像素区域和无像素区域
*/
Mat getMask(const Mat& img)
{
    // 转化为灰度图
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    uchar threshold = 252;
    // mask为中值为255表示为图像，为0表示为丢失的像素
    Mat mask = Mat::zeros(gray.rows, gray.cols, CV_8UC1);

    for(int i = 0; i < gray.rows; i++)
    {
        for(int j = 0; j < gray.cols; j++)
        {
            // 若颜色小于阈值（因图像中缺失的像素为白色即255，小于阈值则说明其为图像），则将mask对应位置置为255
            if(gray.at<uchar>(i, j) < threshold)
                mask.at<uchar>(i, j) = 255;
        }
    }

    Mat res;
    res = fillHole(mask);
    // imshow("mask", res);
    
    // 将mask取反，白色表示为丢失的像素
    res = ~res;

    // 进行三次膨胀
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    Mat dilated;
    dilate(res, dilated, element);
    // imshow("dilated1", dilated);
    dilate(dilated, dilated, element);
    // imshow("dilated2", dilated);
    dilate(dilated, dilated, element);
    // imshow("dilated3", dilated);

    // 进行一次腐蚀
    erode(dilated, res, element);
    // imshow("eroded", res);

    // 经过膨胀和腐蚀使得丢失像素的连通域变得更加完整和平滑，最后取反使得白色（即255）表示为有像素
    res = ~res;

    return res;

    // 是否需要进行膨胀和腐蚀？
    // 进行膨胀和腐蚀后，使得图像边界更
}

/**
 * 绘制直线
*/
void drawLine(Mat& img, const CoordinateDouble& p1, const CoordinateDouble& p2, Scalar color)
{
    line(img, Point(p1.col, p1.row), Point(p2.col, p2.row), color, 1);
}

/**
 * 绘制网格
*/
Mat drawMesh(Mat src, vector<vector<CoordinateDouble>>& mesh, Config config)
{
    Mat img = src.clone();
    // 顶点数
    int mesh_rows = config.mesh_rows;
    int mesh_cols = config.mesh_cols;

    for(int i = 0; i < mesh_rows; i++)
    {
        for(int j = 0; j < mesh_cols; j++)
        {
            // 绘制网格
            if(i != mesh_rows - 1)
            {
                drawLine(img, mesh[i][j], mesh[i + 1][j]); // 往下的线
            }
            if(j != mesh_cols - 1)
            {
                drawLine(img, mesh[i][j], mesh[i][j + 1]); // 往右的线
            }
        }
    }

    return img;
}

/**
 * 按行堆叠矩阵
*/
Eigen::SparseMatrix<double,Eigen::RowMajor> row_stack(const Eigen::SparseMatrix<double,Eigen::RowMajor>& A,
                                                     const Eigen::SparseMatrix<double,Eigen::RowMajor>& B)
{
    Eigen::SparseMatrix<double,Eigen::RowMajor> res(A.rows() + B.rows(), A.cols());
    res.reserve(A.nonZeros() + B.nonZeros()); // 预留稀疏矩阵中非零元素的存储空间，而不实际分配元素。 避免动态分配内存的开销
    res.topRows(A.rows()) = A;
    res.bottomRows(B.rows()) = B;
    return res;
}

/**
 * 将vec转为mesh
*/
vector<vector<CoordinateDouble>> vec2mesh(const VectorXd& vec, const Config& config)
{
    // vec 为 [x,y,x,y,...,x,y]
    int mesh_rows = config.mesh_rows;
    int mesh_cols = config.mesh_cols;
    vector<vector<CoordinateDouble>> mesh(mesh_rows, vector<CoordinateDouble>(mesh_cols));
    for(int i = 0; i < mesh_rows; i++)
    {
        for(int j = 0; j < mesh_cols; j++)
        {
            int index = (i * mesh_cols + j) * 2;
            mesh[i][j] = CoordinateDouble(vec[index+1], vec[index]); // row col : y x
        }
    }
    return mesh;
}

/**
 * 通过mesh获得Vq
*/
VectorXd get_Vq(const vector<vector<CoordinateDouble>>& mesh, int i, int j)
{
    VectorXd Vq(8);
    Vq << mesh[i][j].col, mesh[i][j].row,
        mesh[i][j + 1].col, mesh[i][j + 1].row,
        mesh[i + 1][j].col, mesh[i + 1][j].row,
        mesh[i + 1][j + 1].col, mesh[i + 1][j + 1].row;
    return Vq;
}

/**
 * 扩大mesh
*/
void enlarge_mesh(vector<vector<CoordinateDouble>>& mesh, double s, Config& config)
{
    int mesh_rows = config.mesh_rows;
    int mesh_cols = config.mesh_cols;
    for(int i = 0; i < mesh_rows; i++)
    {
        for(int j = 0; j < mesh_cols; j++)
        {
            mesh[i][j].col *= s;
            mesh[i][j].row *= s;
        }
    }
}

/**
 * 双线性插值获取最终形变后的图像
*/
Vec3f bilinear_interpolation(const Mat& img, double x, double y)
{
    int x1 = static_cast<int>(x);
    int y1 = static_cast<int>(y);
    int x2 = min(x1 + 1, img.cols - 1);
    int y2 = min(y1 + 1, img.rows - 1);

    float a = x - x1;
    float b = y - y1;

    Vec3f pixel = (1 - a) * (1 - b) * img.at<Vec3f>(y1, x1) +
                  a * (1 - b) * img.at<Vec3f>(y1, x2) +
                  (1 - a) * b * img.at<Vec3f>(y2, x1) +
                  a * b * img.at<Vec3f>(y2, x2);

    return pixel;
}