#include <imgRecting.h>
#include <filesystem>

#define DEBUG_TIME // 计时
#define DEBUG_IMGSIZE // 显示原图像大小
// #define DEBUG_MESH // 显示网格的变换

namespace fs = std::filesystem;
using namespace std;

void imgRecting::recting()
{
    if(img.empty())
    {
        cout<<"img is empty"<<endl;
        return;
    }
#ifdef DEBUG_IMGSIZE
    cout<<"原图像大小："<<img.rows<<" "<<img.cols<<endl;
#endif

#ifdef DEBUG_TIME
    // 开始计时
    clock_t start = clock();
#endif

    // 将图像缩放
    Mat scaled_img;
    resize(img,scaled_img,Size(0,0),0.5,0.5); // 缩放比例0.5

    config.rows = scaled_img.rows;
    config.cols = scaled_img.cols;
    config.row_len = (double)(config.rows-1) / config.mesh_quad_rows;
    config.col_len = (double)(config.cols-1) / config.mesh_quad_cols;
    // config = Config(scaled_img.rows,scaled_img.cols); // 配置参数 默认20行20列网格 每个网格有25条线
    Mat mask = getMask(scaled_img);

    // 局部变形
    SeamCarving sc;
    vector<vector<Coordinate>> displacements = sc.get_displacements(scaled_img,mask); // 获取局部位移矩阵
    mesh = sc.place_mesh(scaled_img, config); // 获取网格坐标
    sc.wrap_mesh_back(mesh, displacements, config); // 获取形变后的网格坐标

#ifdef DEBUG_MESH
    // 显示网格
    Mat tmp_img = drawMesh(scaled_img,mesh,config);
    imshow("tmp",tmp_img);
    waitKey(0);
#endif

#ifdef DEBUG_TIME
    // 查看用时
    cout << "局部翘曲获得形变后的mesh用时: " << (clock() - start) / (double)CLOCKS_PER_SEC << "s" << endl;
#endif

    // 全局变形
    GlobalWraping gw(scaled_img, mask, config);
    SparseMatrix<double,RowMajor> shape_mat = gw.get_shape_mat(mesh); // 获取形变矩阵
    cout<<"形变约束矩阵计算完成"<<endl;
    pair<SparseMatrix<double,RowMajor>,VectorXd> pair_dvec_B = gw.get_boundary_mat(mesh); // 获取边界约束
    cout<<"边界约束计算完成"<<endl;

    vector<LineD> lineseg_flatten; // 所有边
    vector<pair<int,double>> id_theta; // 每条边的id以及角度
    vector<double> rotate_theta; // 每条边旋转的角度
    vector<vector<vector<LineD>>> line_seg_mesh = gw.init_line_seg(mesh, lineseg_flatten, id_theta, rotate_theta); // 初始化按方格切割线段
    cout<<"初始化按方格切割线段完成"<<endl;

    SparseMatrix<double,RowMajor> Q_mat = gw.get_Q_mat(mesh); // 获取Q矩阵  将变换应用到全局坐标系中用到的矩阵
    int times = config.times; // 迭代次数

    for(int i=0;i<times;i++)
    {
        cout<<"第"<<i<<"次迭代"<<endl;
#ifdef DEBUG_TIME
        // 查看已用时间
        cout << "现已用时: " << (clock() - start) / (double)CLOCKS_PER_SEC << "s" << endl;
#endif

        /*-----------------------------------固定旋转角度，更新 V-----------------------------------*/
        int n_lines; // 线段个数
        vector<pair<MatrixXd,MatrixXd>> BilinearMat_Vec; // 每条线段的起点和终点的双线性插值权重
        SparseMatrix<double,RowMajor> line_mat = gw.get_line_mat(mesh, line_seg_mesh, rotate_theta, BilinearMat_Vec, n_lines); // 获取线段约束矩阵
        cout<<"获取线段约束矩阵完成  线段个数："<<n_lines<<endl;

        int Nq = config.mesh_quad_rows*config.mesh_quad_cols; // 网格数量
        double lambda_L = 100;
        double lambda_B = INF;
        // 局部网格的形状保留变换应用到全局坐标系中，确保网格在变形过程中的整体形状保持一致
        SparseMatrix<double,RowMajor> shape = sqrt(1.0/Nq) * (shape_mat * Q_mat); // 形变约束
        // 与上同
        SparseMatrix<double,RowMajor> line = sqrt(lambda_L/n_lines) * (line_mat * Q_mat); // 线段约束
        SparseMatrix<double,RowMajor> boundary = sqrt(lambda_B) * pair_dvec_B.first; // 边界约束
        // 上述三个矩阵列数均为 2 * mesh_rows * mesh_cols 即顶点数*2 （坐标有x,y轴，所以*2）

        // 按行堆叠三个矩阵
        SparseMatrix<double,RowMajor> K = row_stack(shape, line);
        SparseMatrix<double,RowMajor> K2 = row_stack(K, boundary); // x * (2*mesh_rows * mesh_cols)

        VectorXd B = pair_dvec_B.second; // 坐标对应的边界值
        // 将其加在 与res相同行数向量的下方
        VectorXd b = VectorXd::Zero(K2.rows()); // x行
        b.tail(B.size()) = sqrt(lambda_B)*B;

        /**
         * 即求解每个坐标与其对应的边界值的欧几里得距离的最小值
         * 需要求解的方程为 min(V) ||K2 * V - b ||²  其中 ||x||² = xT * x 即找到一个 V 使得 K2 * V 和 b 之间的欧几里得距离最小
         * 所以有   K2T * K2 * V - K2T * b = 0  (目标函数对V求导)
        */

        SparseMatrix<double,RowMajor> K2T = K2.transpose();
        SparseMatrix<double,RowMajor> K2_2 = K2T * K2; // 能量的平方 矩阵大小为 (2 * mesh_rows * mesh_cols) * (2 * mesh_rows * mesh_cols)

        b = K2T * b;  // b 变为 (2 * mesh_rows * mesh_cols) * 1 
        // 此时有 K2_2 * V = b    V即为要求解的网格的坐标
        
        // 求解K2_2 * V = b
        // SparseQR<SparseMatrix<double,RowMajor>,Eigen::COLAMDOrdering<int>> solver;
        // SparseLU<SparseMatrix<double,RowMajor>> solver;
        // 使用 SimplicialCholesky 求解对称正定矩阵
        SimplicialCholesky<Eigen::SparseMatrix<double>> solver;
        solver.compute(K2_2);
        if(solver.info()!=Success)
        {
            cout<<"求解K2_2 * V = b出错"<<endl;
            return ;
        }
        VectorXd res = solver.solve(b);
        if(solver.info()!=Success)
        {
            cout<<"求解K2_2 * V = b出错"<<endl;
            return ;
        }

        out_mesh = vec2mesh(res, config); // 将res转换为网格坐标
#ifdef DEBUG_MESH
        // 显示网格
        Mat tmp_img = drawMesh(scaled_img,out_mesh,config);
        imshow("tmp",tmp_img);
        waitKey(0);
#endif

        if(i == times-1)
        {
            break;
        }

        /*-----------------------------------固定 V，更新旋转角度-----------------------------------*/
        int line_id = -1;
        VectorXd thetamGroup = VectorXd::Zero(50); // 50 个 bin
        VectorXd thetamGroupCnt = VectorXd::Zero(50);
        for(int i=0;i<config.mesh_quad_rows;i++)
        {
            for(int j=0;j<config.mesh_quad_cols;j++)
            {
                // 对每一个网格
                vector<LineD> lines = line_seg_mesh[i][j]; // 网格中的所有线段
                if(lines.size()==0)
                {
                    continue;
                }

                int quad_id = i*config.mesh_quad_cols + j;
                VectorXd Vq = get_Vq(out_mesh,i,j);

                for(int k=0;k<lines.size();k++)
                {
                    line_id++;
                    // 待完善，存在错误的线段？
                    
                    pair<MatrixXd,MatrixXd> bilinearMat = BilinearMat_Vec[line_id];
                    Vector2d p1 = bilinearMat.first * Vq; // 插值后的起点
                    Vector2d p2 = bilinearMat.second * Vq; // 插值后的终点

                    double theta = atan((p2[1]-p1[1])/(p2[0]-p1[0])); // 新的角度
                    double delta_theta = theta - id_theta[line_id].second; // 角度变化
                    
                    // 待完善：是否会出错？
                    if(delta_theta>PI/2)
                    {
                        delta_theta -= PI;
                    }
                    else if(delta_theta<-PI/2)
                    {
                        delta_theta += PI;
                    }

                    thetamGroup[id_theta[line_id].first] += delta_theta;
                    thetamGroupCnt[id_theta[line_id].first] += 1;
                }
            }
        }

        // 计算旋转角度平均值
        for(int i=0;i<thetamGroup.size();i++)
        {
            if(thetamGroupCnt[i]!=0)
            {
                thetamGroup[i] /= thetamGroupCnt[i];
            }
            else
            {
                thetamGroup[i] = 0;
            }
        }

        // 更新旋转角度数组
        for(int i=0;i<rotate_theta.size();i++)
        {
            // 旋转角度修改为对应bin的旋转角度均值
            rotate_theta[i] = thetamGroup[id_theta[i].first];
        }
    }

    // 迭代完成，输出
    enlarge_mesh(mesh,2,config);
    enlarge_mesh(out_mesh,2,config);

#ifdef DEBUG_MESH
    tmp_img = drawMesh(img,out_mesh,config);
    imshow("outMesh",tmp_img);
    waitKey(0);
#endif

#ifdef DEBUG_TIME
    cout<<"迭代次数："<<times<<endl;
    cout<<"全景图像矩形化总用时："<<(double)(clock()-start)/CLOCKS_PER_SEC<<"s"<<endl;
#endif

// 以下注释的一段代码可以获得形变后的图像，感觉用的是双线性插值，但是保存下来的图片为什么是黑的？
/*
    Mat out_img = Mat::zeros(img.size(),CV_32FC3);
    Mat out_cnt = Mat::zeros(img.size(),CV_32FC3);
    for(int i=0;i<config.mesh_quad_rows;i++)
    {
        for(int j=0;j<config.mesh_quad_cols;j++)
        {
            VectorXd Vq = get_Vq(out_mesh,i,j);
            VectorXd Vq_old = get_Vq(mesh,i,j);
            double col_len = max({Vq(0),Vq(2),Vq(4),Vq(6)}) - min({Vq(0),Vq(2),Vq(4),Vq(6)});
            double row_len = max({Vq(1),Vq(3),Vq(5),Vq(7)}) - min({Vq(1),Vq(3),Vq(5),Vq(7)});
            double col_step = 1/(col_len*4);
            double row_step = 1/(row_len*4);

            for(double row=0;row<1;row+=row_step)
            {
                for(double col=0;col<1;col+=col_step)
                {
                    double v4w = row * col;
                    double v1w = 1 - row - col + v4w;
                    double v2w = col - v4w;
                    double v3w = row - v4w;
                    MatrixXd mat(2,8);
                    mat << v1w, 0, v2w, 0, v3w, 0, v4w, 0,
                            0, v1w, 0, v2w, 0, v3w, 0, v4w;
                    VectorXd Vq_w = mat * Vq;
                    VectorXd Vq_w_old = mat * Vq_old;
                    int x = Vq_w(0);
                    int y = Vq_w(1);
                    int x_old = Vq_w_old(0);
                    int y_old = Vq_w_old(1);
                    if(x_old>=0 && x_old<img.cols && y_old>=0 && y_old<img.rows)
                    {
                        Vec3b color = img.at<Vec3b>(y_old,x_old);
                        Vec3f color_f = Vec3f(color[0],color[1],color[2]);
                        out_img.at<Vec3f>(y,x) = color_f + out_img.at<Vec3f>(y,x);
                        out_cnt.at<Vec3f>(y,x) += Vec3f(1,1,1);
                    }                    
                }
            }
        }
    }

    Mat save_img = out_img / (out_cnt*255);
    imshow("out_img",save_img);
    waitKey(0);
*/

// #ifdef DEBUG_TIME
//     // 结束计时
//     cout<<"得到最终输出图像总用时："<<(double)(clock()-start)/CLOCKS_PER_SEC<<"s"<<endl;
// #endif

    // imwrite(save_path + "/" + img_path.substr(img_path.find_last_of("/")+1),save_img);
    // cout<<"保存成功"<<endl;
}

void imgRecting::init_gl(string img_path)
{
    // 读取图片
    img = imread(img_path);
    if(img.empty())
    {
        cout<<"img is empty"<<endl;
        return;
    }

    // 初始化GLFW和GLEW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return ;
    }
    window = glfwCreateWindow(img.size().width, img.size().height, "RectanglingImages", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return ;
    }
    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return ;
    }

    getTexture();
}

void imgRecting::getTexture()
{
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Set texture interpolation methods for minification and magnification
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// Set texture clamping method
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // Set incoming texture format to:
	// GL_BGR       for CV_CAP_OPENNI_BGR_IMAGE,
	// GL_LUMINANCE for CV_CAP_OPENNI_DISPARITY_MAP,
	// Work out other mappings as required ( there's a list in comments in main() )
	GLenum inputColourFormat = GL_BGR;
	if (img.channels() == 1)
	{
		inputColourFormat = GL_LUMINANCE;
	}

    // Create the texture
	glTexImage2D(GL_TEXTURE_2D,     // Type of texture
		0,                 // Pyramid level (for mip-mapping) - 0 is the top level
		GL_RGB,            // Internal colour format to convert to
		img.cols,          // Image width  i.e. 640 for Kinect in standard mode
		img.rows,          // Image height i.e. 480 for Kinect in standard mode
		0,                 // Border width in pixels (can either be 1 or 0)
		inputColourFormat, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
		GL_UNSIGNED_BYTE,  // Image data type
		img.ptr());        // The actual image data itself
}

void imgRecting::render()
{
    // 绘制变形后的网格
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity(); // 重置视图矩阵

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureID);
    // glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f); // 左下角
    // glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, -1.0f);  // 右下角
    // glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 1.0f);   // 右上角
    // glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, 1.0f);  // 左上角

    // 确保网格数据有效
    glBegin(GL_QUADS);

    for (int i = 0; i < config.mesh_rows - 1; ++i) {
        for (int j = 0; j < config.mesh_cols - 1; ++j) {
            // 获取变形后的网格顶点和纹理坐标
            const CoordinateDouble& v1 = out_mesh[i][j];     // 变形后的网格顶点1
            const CoordinateDouble& v2 = out_mesh[i][j + 1]; // 变形后的网格顶点2
            const CoordinateDouble& v3 = out_mesh[i + 1][j + 1]; // 变形后的网格顶点3
            const CoordinateDouble& v4 = out_mesh[i + 1][j]; // 变形后的网格顶点4

            const CoordinateDouble& uv1 = mesh[i][j];     // 对应的纹理坐标1
            const CoordinateDouble& uv2 = mesh[i][j + 1]; // 对应的纹理坐标2
            const CoordinateDouble& uv3 = mesh[i + 1][j + 1]; // 对应的纹理坐标3
            const CoordinateDouble& uv4 = mesh[i + 1][j]; // 对应的纹理坐标4

            // 绘制四边形
            glTexCoord2f(uv1.col / img.cols, uv1.row / img.rows);
            glVertex2f(v1.col / img.cols * 2.0f - 1.0f, 1.0f - v1.row / img.rows * 2.0f);

            glTexCoord2f(uv2.col / img.cols, uv2.row / img.rows);
            glVertex2f(v2.col / img.cols * 2.0f - 1.0f, 1.0f - v2.row / img.rows * 2.0f);

            glTexCoord2f(uv3.col / img.cols, uv3.row / img.rows);
            glVertex2f(v3.col / img.cols * 2.0f - 1.0f, 1.0f - v3.row / img.rows * 2.0f);

            glTexCoord2f(uv4.col / img.cols, uv4.row / img.rows);
            glVertex2f(v4.col / img.cols * 2.0f - 1.0f, 1.0f - v4.row / img.rows * 2.0f);
        }
    }

    glEnd();
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void imgRecting::display()
{
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);

        render();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

void imgRecting::save_img(string path)
{
    // 获取当前时间
    time_t now = time(0);
    char* dt = ctime(&now);

    // 判断目标路径是否存在
    if(!fs::exists(path))
    {
        fs::create_directories(path);
    }

    // 保存图片
    string img_path = path + "/" + dt + ".jpg";


    int width = img.cols;
    int height = img.rows;
    vector<unsigned char> buffer(width * height * 3);
    // 读取帧缓冲区内容到 buffer 中
    glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, buffer.data());
    // 将像素数据转换为 OpenCV Mat
    Mat img_save(height, width, CV_8UC3, buffer.data());

    // 由于 OpenGL 原点在左下角，OpenCV 原点在左上角，需要翻转图像
    cv::flip(img_save, img_save, 0);

    // 保存图像到文件
    cv::imwrite(img_path, img_save);
}