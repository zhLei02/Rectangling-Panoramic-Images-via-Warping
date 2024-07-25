#include <GlobalWraping.h>

// #define DEBUG_LSD       // 画出LSD检测出的直线
// #define DEBUG_SEGLINE   // 画出使用mesh切割线段的结果
// #define DEBUG_LINEMAT   // 显示出每个方格线段数目以及总数目

/**
 * 构造函数
*/
GlobalWraping::GlobalWraping(Mat& img, Mat& mask, Config& config)
{
    this->img = img;
    this->mask = mask;
    this->config = config;
}


/**
 * 获取Shape Presevation矩阵
*/
Eigen::SparseMatrix<double,Eigen::RowMajor> GlobalWraping::get_shape_mat(const vector<vector<CoordinateDouble>>& mesh)
{
    // 网格顶点数以及边长数
    int mesh_rows = config.mesh_rows;
    int mesh_cols = config.mesh_cols;
    int mesh_quad_rows = config.mesh_quad_rows;
    int mesh_quad_cols = config.mesh_quad_cols;

    // 构造Shape Preservation矩阵
    // 一个大的稀疏矩阵记录每一个小网格构造一个8*8的矩阵存储论文中公式 (Aq * ((AqT * Aq)-1) * AqT - I) 的结果记为coeff，其中Aq为 8*4，故coeff为8*8
    Eigen::SparseMatrix<double,Eigen::RowMajor> shape_mat(8.0*mesh_quad_rows*mesh_quad_cols, 8.0*mesh_quad_rows*mesh_quad_cols);

    for(int i=0; i<mesh_quad_rows; i++)
    {
        for(int j=0; j<mesh_quad_cols; j++)
        {
            // 四个顶点         Vq的顺序为 左上，右上，左下，右下
            CoordinateDouble p0 = mesh[i][j];       // 左上
            CoordinateDouble p1 = mesh[i][j+1];     // 右上
            CoordinateDouble p2 = mesh[i+1][j];     // 左下
            CoordinateDouble p3 = mesh[i+1][j+1];   // 右下
            
            // 构造Aq
            Eigen::MatrixXd Aq(8,4);
            Aq << p0.col, -p0.row, 1, 0,
                  p0.row, p0.col, 0, 1,
                  p1.col, -p1.row, 1, 0,
                  p1.row, p1.col, 0, 1,
                  p2.col, -p2.row, 1, 0,
                  p2.row, p2.col, 0, 1,
                  p3.col, -p3.row,1, 0,
                  p3.row, p3.col, 0, 1;
            // AqT
            Eigen::MatrixXd AqT = Aq.transpose();
            // ((AqT * Aq)-1)
            Eigen::MatrixXd AqT_Aq__inverse = (AqT*Aq).inverse();
            // coeff
            Eigen::MatrixXd coeff = Aq * AqT_Aq__inverse * AqT - Eigen::MatrixXd::Identity(8,8);

            int left_top_row = 8 * (i*mesh_quad_cols + j); // 每一个网格占用8行，故第i*mesh_quad_cols+j个网格的左上角行坐标为8*(i*mesh_quad_cols+j)
            for(int k=0; k<8; k++)
            {
                for(int l=0; l<8; l++)
                {
                    shape_mat.insert(left_top_row+k, left_top_row+l) = coeff(k,l);
                    // 例
                    // 第一个矩阵插入为 (0,0)~(7,7)
                    // 第二个矩阵插入为 (8,8)~(15,15)
                }
            }
        }
    }
    shape_mat.makeCompressed(); // 压缩稀疏矩阵 优化稀疏矩阵的存储和计算效率 通过将矩阵从插入模式转换为压缩模式，可以减少内存占用并加快后续的矩阵运算和求解过程
    return shape_mat;
}

/**
 * 构造矩阵 Q 其大小为 (8*mesh_quad_rows*mesh_quad_cols,2*mesh_rows*mesh_cols)
 * 8*mesh_quad_rows*mesh_quad_cols 表示存在有mesh_quad_rows*mesh_quad_cols个网格，每个网格有4个顶点，8个坐标
 * 2*mesh_rows*mesh_cols 表示有mesh_rows*mesh_cols个顶点，每个顶点有2个坐标
 * 通过 Q*V 即可获得某个网格的顶点坐标，其中V表示所有顶点的一个一维坐标向量 其大小为(2*mesh_rows*mesh_cols) * 1的向量 [x0,y0,x1,y1,....,xn,yn]
*/
Eigen::SparseMatrix<double,Eigen::RowMajor> GlobalWraping::get_Q_mat(const vector<vector<CoordinateDouble>>& mesh)
{
    // 网格参数
    int mesh_rows = config.mesh_rows;
    int mesh_cols = config.mesh_cols;
    int mesh_quad_rows = config.mesh_quad_rows;
    int mesh_quad_cols = config.mesh_quad_cols;

    // 稀疏矩阵Q
    Eigen::SparseMatrix<double,Eigen::RowMajor> Q_mat(8*mesh_quad_rows*mesh_quad_cols, 2*mesh_rows*mesh_cols);

    for(int i=0; i<mesh_quad_rows; i++)
    {
        for(int j=0; j<mesh_quad_cols; j++)
        {
            int quad_id = 8 * (i*mesh_quad_cols + j); // 每个quad的起始顶点坐标行数
            int vertex_id = 2 * (i*mesh_cols + j); // 该quad对应的左上角顶点id
            Q_mat.insert(quad_id, vertex_id) = 1; // 左上顶点x坐标
            Q_mat.insert(quad_id+1, vertex_id+1) = 1; // 左上顶点y坐标
            Q_mat.insert(quad_id+2, vertex_id+2) = 1; // 右上顶点x坐标
            Q_mat.insert(quad_id+3, vertex_id+3) = 1; // 右上顶点y坐标
            Q_mat.insert(quad_id+4, vertex_id+2*mesh_cols) = 1; // 左下顶点x坐标
            Q_mat.insert(quad_id+5, vertex_id+2*mesh_cols+1) = 1; // 左下顶点y坐标
            Q_mat.insert(quad_id+6, vertex_id+2*mesh_cols+2) = 1; // 右下顶点x坐标
            Q_mat.insert(quad_id+7, vertex_id+2*mesh_cols+3) = 1; // 右下顶点y坐标
        }
    }

    Q_mat.makeCompressed();
    return Q_mat;
}


/**
 * 获取边界约束矩阵
*/
pair<Eigen::SparseMatrix<double,Eigen::RowMajor>,Eigen::VectorXd> GlobalWraping::get_boundary_mat(const vector<vector<CoordinateDouble>>& mesh)
{
    // 图像参数
    int rows = config.rows;
    int cols = config.cols;
    // 网格参数
    int mesh_rows = config.mesh_rows;
    int mesh_cols = config.mesh_cols;
    int vertex_num = mesh_rows*mesh_cols; // 网格顶点数

    /**
     * dvec：这是一个表示边界权重的向量。值为1的元素表示对应顶点在边界上，并且该顶点的位置被固定。值为0的元素表示该顶点的位置不受固定。
     * B：这是一个表示边界位置的向量。它的元素值表示对应顶点在图像边界上的具体坐标值。
    */
    VectorXd dvec = VectorXd::Zero(vertex_num*2); // 用一个向量表示所有顶点 dvec:[x0,y0,x1,y1,....,xn,yn]
    VectorXd B = VectorXd::Zero(vertex_num*2); // 边界向量，存储边界大小
    // 处理左边界   每一个左边界的x坐标处dvec置为1，B置为0，表示此处为边界，需要加入约束
    for(int i=0; i<vertex_num*2; i+= mesh_cols*2)
    {
        dvec(i) = 1;
        B(i) = 0;
    }
    // 处理右边界 第一个右边界点为第 mesh_cols-1 个，故其x坐标为2*mesh_cols-2
    for(int i=2*mesh_cols-2; i<vertex_num*2; i+= mesh_cols*2)
    {
        dvec(i) = 1;
        B(i) = cols-1;
    }
    // 处理上边界 只与y坐标有关
    for(int i=1;i<mesh_cols*2;i+=2)
    {
        dvec(i) = 1;
        B(i) = 0;
    }
    // 处理下边界 第一个下边界点为第 mesh_cols*(mesh_rows-1) 个，故其y坐标为 2*mesh_cols*(mesh_rows-1)+1
    for(int i=2*mesh_cols*(mesh_rows-1)+1;i<vertex_num*2;i+=2)
    {
        dvec(i) = 1;
        B(i) = rows-1;
    }

    // 生成对应的稀疏对角矩阵
    SparseMatrix<double,Eigen::RowMajor> diag(vertex_num*2, vertex_num*2);
    for(int i=0; i<vertex_num*2; i++)
        diag.insert(i,i) = dvec(i);
    diag.makeCompressed();
    return make_pair(diag,B);
}

/**
 * 判断line在图像内部
*/
bool GlobalWraping::line_in_mask(const LineD& line)
{
    if(mask.at<uchar>(line.p1.row,line.p1.col) == haspixel || mask.at<uchar>(line.p2.row,line.p2.col) == haspixel)
        return true;
    return false;
}

/**
 * 调用lsd算法进行线段检测
 * lsd算法介绍： （计算前一般使用高斯降采样降低图像的锯齿效应）
 * 1. 计算每个像素点处的梯度方向以及幅度   并   在一定容忍度（22.5°较好）下找到具有相同梯度方向的连通区域作为可能的线段区域
 * 2. 用线段区域的主轴表示矩形的主轴，用一个矩形覆盖整个区域  
 *      （因像素处理顺序对结果有影响，采用贪心优先处理较高梯度幅值的像素，为实现线性时间复杂的的排序，
 *      将像素点的梯度幅值(0~max)分为了1024个bin，逐个bin处理。     
 *      因其为贪心处理，故在此次图像中有像素和无像素的边界处的梯度变化值一般较大，导致边界处存在线段，一般不需要维护该边界的平行关系，可以进行特殊处理
 * 3. 找到所有矩形内与矩形主轴在一定容忍度下的像素点称之为对齐的像素点
 * 4. 根据矩形中像素点的个数和对齐像素点的个数关系以验证该矩形区域是否可以作为直线分割
*/
vector<LineD> GlobalWraping::lsd_detect()
{
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY); // 转化为灰度图
    double* image = new double[gray.rows*gray.cols]; // 用double数组存储灰度图（lsd的要求）
    for(int i=0; i<gray.rows; i++)
        for(int j=0; j<gray.cols; j++)
            image[i*gray.cols+j] = gray.at<uchar>(i,j);
    
    // 调用lsd算法获取所有线段
    vector<LineD> lines;
    double* out; // lsd输出
    int n; // 检测到的线段个数
    out = lsd(&n,image,gray.cols,gray.rows); // 调用lsd算法进行线段检测

#ifdef DEBUG_LSD
    Mat tmp = img.clone();
#endif

    for(int i=0;i<n;i++)
    {
        CoordinateDouble start = CoordinateDouble(out[i*7+1],out[i*7+0]);
        CoordinateDouble end = CoordinateDouble(out[i*7+3],out[i*7+2]);
        LineD line = LineD(start,end);

#ifdef DEBUG_LSD
        drawLine(tmp,start,end);
#endif
        // 实际上无需判断，检查到的线段一定在图像内部，图像外部均为同一颜色。
        // 此处仍加上该判断，增加一定的算法鲁棒性
        if(line_in_mask(line))
        {
            lines.push_back(line);
        }
        else
        {
            // 基本上没有
            cout<<"line out of mask"<<endl;
        }
        // 此处暂时不处理有无像素边界处的直线
        // 应该无需处理，后续线段切割过程中，这种线条不在mesh内部，未切割，只不过会降低一定的速度，但就这么几条线，对于cpp来说，应该可以忽略不计
    }

#ifdef DEBUG_LSD
    imshow("line",tmp);
    waitKey(0);
#endif

    // 释放内存
    delete[] image;
    delete[] out;
    return lines;
}

/*------------------------------用于将线段切割到各网格时的工具函数------------------------------*/

// 判断点是否在四边形内
bool GlobalWraping::is_in_quad(const CoordinateDouble& p, const CoordinateDouble& lefttp, const CoordinateDouble& righttp,
                                 const CoordinateDouble& leftbt, const CoordinateDouble& rightbt)
{
    // 该点需要在左边界右边，上边界下面，右边界左边，下边界上边

    // 根据叉乘判断方向 例如顶点由左上开始顺时针为 A、B、C、D， 那么 AB × AP < 0,则 P 在AB的顺时针方向
    auto cross = [](CoordinateDouble A, CoordinateDouble B, CoordinateDouble P)
    {
        // 叉乘计算公式  u × v = ux vy - uy vx
        // AB : x: (B.col-A.col) y: (B.row-A.row)
        // AP : x: (P.col-A.col) y: (P.row-A.row)
        return (B.col-A.col)*(P.row-A.row) - (B.row-A.row)*(P.col-A.col);
    }; 

    CoordinateDouble points[4] = {lefttp,righttp,rightbt,leftbt};

    // 待完善：为什么是小于0不在内部啊？
    // for(int i=0; i<4; i++)
    // {
    //     if(cross(points[i],points[(i+1)%4],p) < 0) // 不在四边形内 为什么小于0是不在四边形内?
    //         return false;
    // }
    // return true; // 在四边形内或边界

    double ans[4] = {0,0,0,0};
    for(int i=0; i<4; i++)
    {
        ans[i] = cross(points[i],points[(i+1)%4],p);
    }
    if((ans[0] < 0 && ans[1] < 0 && ans[2] < 0 && ans[3] < 0)||(ans[0] > 0 && ans[1] > 0 && ans[2] > 0 && ans[3] > 0))
        return true;
    return false;
}

// 获取两条线段相交的点
bool GlobalWraping::get_intersection(const LineD& l1, const LineD& l2, CoordinateDouble& intersection)
{
    // 俩条线段，四个点，8个坐标如下
    double x1 = l1.p1.col,y1 = l1.p1.row,x2 = l1.p2.col,y2 = l1.p2.row;
    double x3 = l2.p1.col,y3 = l2.p1.row,x4 = l2.p2.col,y4 = l2.p2.row;

    // 通过求解两线段相交的方程组得到交点坐标
    double denom = (x2-x1)*(y4-y3) - (x4-x3)*(y2-y1); // 分母
    if(denom == 0) // 相交点不存在
        return false;
    
    double t1 = ((x3-x1)*(y4-y3)-(x4-x3)*(y3-y1)) / denom;
    double t2 = ((y2-y1)*(x3-x1)-(y3-y1)*(x2-x1)) / denom;

    if(t1 >= 0 && t1 <= 1 && t2 >= 0 && t2 <= 1) // 相交点存在
    {
        intersection = CoordinateDouble(y1+t1*(y2-y1),x1+t1*(x2-x1));
        return true;
    }

    return false;
}

// 获取一条线与四边形相交的点的集合
vector<CoordinateDouble> GlobalWraping::get_intersections(const LineD& line, const CoordinateDouble& lefttp,
         const CoordinateDouble& righttp, const CoordinateDouble& leftbt, const CoordinateDouble& rightbt)
{
    vector<CoordinateDouble> intersections;
    LineD edge[4] = {
        LineD(lefttp,righttp),
        LineD(righttp,rightbt),
        LineD(rightbt,leftbt),
        LineD(leftbt,lefttp)
    };
    // 判断线段与四边形的四条边是否有交点，若有，加入交点集合中
    for(int i=0; i<4; i++)
    {
        CoordinateDouble intersection;
        if(get_intersection(line,edge[i],intersection))
            intersections.push_back(intersection);
    }
    return intersections;
}

// 将切割后得到的线段三维数组展平
vector<LineD> GlobalWraping::flatten_lines(const vector<vector<vector<LineD>>>& line_seg)
{
    vector<LineD> lines_flatten;
    int mesh_quad_rows = config.mesh_quad_rows;
    int mesh_quad_cols = config.mesh_quad_cols;
    for(int i=0; i<mesh_quad_rows; i++)
        for(int j=0; j<mesh_quad_cols; j++)
            for(int k=0; k<line_seg[i][j].size(); k++)
                lines_flatten.push_back(line_seg[i][j][k]);
    return lines_flatten;
}

/*--------------------------------------------END--------------------------------------------*/

/**
 * 将线段分割到各个网格中
*/
vector<vector<vector<LineD>>> GlobalWraping::segline_inquad(const vector<LineD>& lines, const vector<vector<CoordinateDouble>>& mesh)
{
    // 网格行列数
    int mesh_quad_rows = config.mesh_quad_rows;
    int mesh_quad_cols = config.mesh_quad_cols;
#ifdef DEBUG_SEGLINE
    Mat tmp = img.clone();
#endif

    vector<vector<vector<LineD>>> line_seg_mesh; // 某行，某列的线段集合
    for(int i=0; i<mesh_quad_rows; i++)
    {
        vector<vector<LineD>> rowVec;
        for(int j=0; j<mesh_quad_cols; j++)
        {
            // 对于网格 mesh[i][j]
            CoordinateDouble p1 = mesh[i][j]; // 左上顶点
            CoordinateDouble p2 = mesh[i][j+1]; // 右上顶点
            CoordinateDouble p3 = mesh[i+1][j]; // 左下顶点
            CoordinateDouble p4 = mesh[i+1][j+1];  // 右下顶点

            // 存储线段集合
            vector<LineD> line_seg_quad;
            for(auto line : lines)
            {
                bool p1_in_quad = is_in_quad(line.p1,p1,p2,p3,p4);
                bool p2_in_quad = is_in_quad(line.p2,p1,p2,p3,p4);
                if(p1_in_quad && p2_in_quad) // 线段在网格内
                {
                    line_seg_quad.push_back(line);
                }
                else if(p1_in_quad) // 线段起点在网格内，终点不在，则存在交点
                {
                    vector<CoordinateDouble> intersections = get_intersections(line,p1,p2,p3,p4);
                    if(intersections.size() == 1)
                    {
                        line_seg_quad.push_back(LineD(line.p1,intersections[0]));
                    }
                }
                else if(p2_in_quad) // 线段终点在网格内，起点不在，则存在交点
                {
                    vector<CoordinateDouble> intersections = get_intersections(line,p1,p2,p3,p4);
                    if(intersections.size() == 1)
                    {
                        line_seg_quad.push_back(LineD(intersections[0],line.p2));
                    }
                }
                else // 线段起点和终点都不在网格内，则要么存在两个交点，需切割，要么存在一个交点即相切或不存在交点，无需切割
                {
                    vector<CoordinateDouble> intersections = get_intersections(line,p1,p2,p3,p4);
                    if(intersections.size() == 2)
                    {
                        // 此处线段的方向是什么样的呢？
                        // 线段的起点和终点并不重要，关键在于计算与x轴夹角，此时无论起点终点关系如何，得到的夹角值是一样的
                        line_seg_quad.push_back(LineD(intersections[0],intersections[1]));
                    }
                }
            }
#ifdef DEBUG_SEGLINE
                drawLine(tmp,p1,p2,Scalar(0,0,255)); // 上边界
                drawLine(tmp,p1,p3,Scalar(0,0,255)); // 左边界
                drawLine(tmp,p2,p4,Scalar(0,0,255)); // 右边界
                drawLine(tmp,p3,p4,Scalar(0,0,255)); // 下边界
                for(auto l:line_seg_quad)
                {
                    drawLine(tmp,l.p1,l.p2);
                }
                imshow("line_seg",tmp);
                waitKey(0);
#endif
            rowVec.push_back(line_seg_quad);
        }
        line_seg_mesh.push_back(rowVec);
    }

    return line_seg_mesh;
}

/**
 * 初始化线段分割,将具有相近倾斜角度的线段分配到一个集合中
 * line_seg_flatten: 所有线段集合
 * id_theta：存储每条线段对应的bin_id和初始角度
 * rotate_theta：存储每条线段的旋转角度
*/
vector<vector<vector<LineD>>> GlobalWraping::init_line_seg(const vector<vector<CoordinateDouble>>& mesh,
             vector<LineD>& line_seg_flatten, vector<pair<int,double>>& id_theta, vector<double>& rotate_theta)
{
    double theta_bin = PI / 49; // 50个角度区间  (-PI/2 + PI/2)/theta_bin = 0   (PI/2 + PI/2)/theta_bin = 49  0~49
    
    // 第一步：使用LSD算法进行直线检测
    vector<LineD> lines = lsd_detect();

    // 第二步：根据mesh将线段切割分到各网格中
    vector<vector<vector<LineD>>> line_seg_mesh = segline_inquad(lines,mesh);

    // 第三步：将具有相近倾斜角度的线段分配到一个集合中
    line_seg_flatten = flatten_lines(line_seg_mesh);
    for(auto& line:line_seg_flatten)
    {
        /*出现错误，atan和atan2不一样，atan2对象限敏感，值域范围[-pi, pi]；atan对象限不敏感，值域范围为[-pi/2, pi/2]*/
        double theta = atan((line.p2.row - line.p1.row)/(line.p2.col - line.p1.col));
        // cout<<"theta: "<<theta<<endl;
        int bin_id = round((theta + PI / 2) / theta_bin);
        // cout<<"bin_id: "<<bin_id<<endl;
        assert(bin_id >= 0 && bin_id < 50);
        id_theta.push_back(make_pair(bin_id,theta)); // 存储每条线段对应的bin_id和原直线角度
        rotate_theta.push_back(0); // 初始旋转角度设定为0，即不旋转
    }

    return line_seg_mesh;
}



/*-------------------------------用于计算线段约束矩阵的工具函数-------------------------------*/

// 计算双线性插值权重  p1,p2,p3,p4为四边形左上顶点开始，按顺时针顺序给出
BilinearWeights GlobalWraping::get_bilinear_weights(const CoordinateDouble& p, const CoordinateDouble& a,
                 const CoordinateDouble& b, const CoordinateDouble& c, const CoordinateDouble& d)
{
    BilinearWeights weights(-1.0,-1.0);

    CoordinateDouble e = b-a;
    CoordinateDouble f = d-a;
    CoordinateDouble g = a-b+c-d;
    CoordinateDouble h = p-a;

    auto cross = [](const CoordinateDouble& a, const CoordinateDouble& b)
    {
        return a.col * b.row - a.row * b.col;
    };

    double k2 = cross(g, f);
    double k1 = cross(e, f) + cross(h, g);
    double k0 = cross(h, e);

    // 如果边是平行的，这是一个线性方程
    if (abs(k2) < 0.001) {
        weights.s = (h.col * k1 + f.col * k0) / (e.col * k1 - g.col * k0);
        weights.t = -k0 / k1;
    }
    else { // 否则，这是一个二次方程
        double w = k1 * k1 - 4.0 * k0 * k2;
        if (w < 0.0) return weights; // 无解，返回默认值

        w = sqrt(w);
        double ik2 = 0.5 / k2;
        double v = (-k1 - w) * ik2;
        double u = (h.col - f.col * v) / (e.col + g.col * v);

        if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) {
            v = (-k1 + w) * ik2;
            u = (h.col - f.col * v) / (e.col + g.col * v);
        }
        weights.s = u;
        weights.t = v;
    }

    return weights;
}

/**
 * 将双线性插值权重转化为矩阵 f(s,t) = (1-s)(1-t)fp1 + s(1-t)fp2 + t(1-s)fp3 + s*t*fp4
 * 矩阵为 2*8 ，其中 v1 = 1 - s - t + st、 v2 = s - st、v3 = t - st、v4 = st
 * 矩阵样式如下
 * v1   0   v2   0   v3   0   v4   0
 * 0   v1   0   v2   0   v3   0   v4
 * Vq 为 8*1 矩阵，即四边形四个顶点坐标 [x0,y0,x1,y1,x2,y2,x3,y3] 顺序为 左上、右上、左下、右下
*/
MatrixXd GlobalWraping::bilinearW2Mat(const BilinearWeights& w)
{
    MatrixXd mat(2,8);
    double v4 = w.s * w.t; // st
    double v1 = 1 - w.s - w.t + v4; // 1-s-t+st
    double v2 = w.s - v4; // s-st
    double v3 = w.t - v4; // t-st
    mat << v1, 0, v2, 0, v3, 0, v4, 0,
            0, v1, 0, v2, 0, v3, 0, v4;
    return mat;
}


/*-------------------------------------------END-------------------------------------------*/

/**
 * 线段约束矩阵
*/
Eigen::SparseMatrix<double,Eigen::RowMajor> GlobalWraping::get_line_mat(const vector<vector<CoordinateDouble>>& mesh,
    const vector<vector<vector<LineD>>>& line_seg_mesh, const vector<double>& rotate_theta,
     vector<pair<MatrixXd, MatrixXd>>& BilinearMat_Vec, int& n_lines)
{
    // 用于性能优化 因矩阵扩展不断重复分配内存空间和拷贝，可能导致程序运行缓慢
    int init_Ce_rows = 2*config.lines_per_quad; // 假设开始分配50行空间，即每个方格25条线段
    int init_lineMat_rows = init_Ce_rows/2*config.mesh_quad_rows*config.mesh_quad_cols;

    // 图片参数
    int rows = config.rows;
    int cols = config.cols;
    // 网格数
    int mesh_quad_rows = config.mesh_quad_rows;
    int mesh_quad_cols = config.mesh_quad_cols;

    // 线段约束矩阵
    Eigen::SparseMatrix<double,Eigen::RowMajor> line_mat(init_lineMat_rows,8*mesh_quad_rows*mesh_quad_cols);
    int cur_lineMat_rows = 0;
    int line_id = -1; // 记录线段id

    for(int i=0; i<mesh_quad_rows; i++)
    {
        for(int j=0; j<mesh_quad_cols; j++)
        {
            // 对每一个方格的所有线段
            vector<LineD> lines = line_seg_mesh[i][j];
            int quad_id = i * mesh_quad_cols + j;

            if(lines.size() == 0) // 不存在线段
                continue;
            
            MatrixXd Ce_rowStack(init_Ce_rows,8); // 按行方向堆叠所有线段的Ce矩阵
            int cur_Ce_rows = 0;
            
            const CoordinateDouble& p1 = mesh[i][j]; // 左上顶点
            const CoordinateDouble& p2 = mesh[i][j+1]; // 右上顶点
            const CoordinateDouble& p3 = mesh[i+1][j]; // 左下顶点
            const CoordinateDouble& p4 = mesh[i+1][j+1]; // 右下顶点

            for(int k=0; k<lines.size(); k++)
            {
                line_id++; // 线段id
                LineD line = lines[k];
                CoordinateDouble& start = line.p1; // 起点
                CoordinateDouble& end = line.p2; // 终点

                // 因网格的边不一定平行，故采用逆双线性插值计算权重，再求得
                BilinearWeights start_weights = get_bilinear_weights(start,p1,p2,p4,p3);
                BilinearWeights end_weights = get_bilinear_weights(end,p1,p2,p4,p3);
                // 得到权重后，转化为2*8矩阵，使得矩阵与Vq相乘即可得到对应的方向向量值
                MatrixXd start_weights_mat = bilinearW2Mat(start_weights);
                MatrixXd end_weights_mat = bilinearW2Mat(end_weights);
                // 获取Vq
                VectorXd Vq = get_Vq(mesh,i,j);
                // 得到双线性插值后的起点和终点与原起点和中点的插值
                Vector2d diff1 = start_weights_mat * Vq - Vector2d(start.col,start.row);
                Vector2d diff2 = end_weights_mat * Vq - Vector2d(end.col,end.row);

                if(diff1.norm() > 0.0001 || diff2.norm() > 0.0001)
                {
                    cout<< "差距过大，双线性插值出现错误？" <<endl;
                    continue;
                }

                double thetam = rotate_theta[line_id]; // 直线的当前旋转角度
                BilinearMat_Vec.emplace_back(start_weights_mat, end_weights_mat); // 存储当前直线的双线性插值矩阵
                Matrix2d R = Eigen::Rotation2Dd(thetam).toRotationMatrix(); // 旋转矩阵
                // Matrix2d R;
                // R << cos(thetam), -sin(thetam), sin(thetam), cos(thetam);
                MatrixXd ehat(2,1);
                ehat << line.p2.col - line.p1.col, line.p2.row - line.p1.row;
                MatrixXd ehatT_ehat_inverse = (ehat.transpose() * ehat).inverse();
                MatrixXd C = R * ehat * ehatT_ehat_inverse * ehat.transpose() * R.transpose() - MatrixXd::Identity(2,2); // 论文中的矩阵C
                MatrixXd Ce = C * (end_weights_mat - start_weights_mat); // C*e

                if(cur_Ce_rows + Ce.rows() > Ce_rowStack.rows()) // 当前行数不够，扩展行数
                {
                    init_Ce_rows += 50; // 每次扩展50行，即25条线段
                    Ce_rowStack.conservativeResize(init_Ce_rows,8);
                }
                Ce_rowStack.block(cur_Ce_rows, 0, Ce.rows(), Ce.cols()) = Ce;
                cur_Ce_rows += Ce.rows();
            }
            // 调整大小为实际使用的行数
            Ce_rowStack.conservativeResize(cur_Ce_rows,8);
#ifdef DEBUG_LINEMAT
            cout<< "Ce_rowStack_Size: "<<cur_Ce_rows<<endl;
#endif

            // 将Ce_rowStack矩阵按照主对角线存储在line_mat中
            if(cur_lineMat_rows + Ce_rowStack.rows() > line_mat.rows())
            {
                init_lineMat_rows += 20*config.mesh_quad_rows*config.mesh_quad_cols;
                line_mat.conservativeResize(init_lineMat_rows, 8 * config.mesh_quad_rows * config.mesh_quad_cols);
            }
            int lefttop_row = cur_lineMat_rows;
            int lefttop_col = quad_id * 8;
            cur_lineMat_rows += Ce_rowStack.rows();
            for(int i=0; i<Ce_rowStack.rows(); i++)
            {
                for(int j=0; j<Ce_rowStack.cols(); j++)
                {
                    line_mat.insert(lefttop_row + i, lefttop_col + j) = Ce_rowStack(i,j);
                }
            }
        }
    }
    n_lines = line_id + 1;
    line_mat.conservativeResize(cur_lineMat_rows, 8 * config.mesh_quad_rows * config.mesh_quad_cols);
#ifdef DEBUG_LINEMAT
    cout<< "line_mat_Size: "<<cur_lineMat_rows<<endl;
#endif
    line_mat.makeCompressed(); // 压缩
    return line_mat;
}



