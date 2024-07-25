#include <SeamCarving.h>

// #define DEBUG // 展示最终矩形化的图片结果
// #define DEBUG_ENERGY // 展示SeamCarving算法的能量图计算结果，即进行边缘检测后的结果（梯度计算）
// #define DEBUG_SEAM // 展示找到的seam
// #define DEBUG_BORDER // 展示找到的最长边界段

// 判断mask的(row,col)处是否为透明的
bool SeamCarving::is_transparent(const Mat& mask, int row,int col)
{
    if(mask.at<uchar>(row,col)==haspixel) // 有像素，不透明
        return false;
    return true; // 没有像素，透明
}

// 初始化偏移量
void SeamCarving::init_displacement(vector<vector<Coordinate>>& displacement, int rows, int cols)
{
    displacement = vector<vector<Coordinate>>(rows,vector<Coordinate>(cols,Coordinate()));
}

/**
 * 找到最长的边界段，返回起始坐标和终点坐标
 * mask表示被遮住的区域，当不为0时表示被遮住
*/
pair<int,int> SeamCarving::choose_longestborder(const Mat& img, const Mat& mask, Border& direction)
{
    int rows = img.rows;
    int cols = img.cols;

    int maxL = 0; // 最大长度
    int s = 0, e = 0; // 起始坐标和终点坐标

    // 帮助函数：用于检测是否需要更新记录
    auto check_longer = [&](int lengh,int start,int end,Border _direction)
    {
        if(lengh>maxL)
        {
            maxL = lengh;
            s = start;
            e = end;
            direction = _direction;
        }
    };
    

    int tL = 0;
    int ts = 0, te = 0;
    bool isCounting = false;
    // 帮助函数：增加计数
    auto longerL = [&]()
    {
        te++;
        tL++;
        isCounting = true;
    };

    // 左侧边界
    for(int i=0;i<rows;i++)
    {
        if(!is_transparent(mask,i,0) || i==rows-1) // 非透明 或 已到结尾   停止计数
        {
            if(isCounting)
            {
                if(is_transparent(mask,i,0))
                {
                    tL++;
                    te++;
                }
                check_longer(tL,ts,te,Border::LEFT);
            }
            isCounting = false;
            ts = te = i+1;
            tL = 0;
        }
        else // 透明
        {
            longerL();
        }
    }

    // 右侧边界
    tL=ts=te=0;
    isCounting = false;
    for(int i=0;i<rows;i++)
    {
        if(!is_transparent(mask,i,cols-1) || i==rows-1) // 非透明 或 已到结尾   停止计数
        {
            if(isCounting)
            {
                if(is_transparent(mask,i,cols-1))
                {
                    tL++;
                    te++;
                }
                check_longer(tL,ts,te,Border::RIGHT);
            }
            isCounting = false;
            ts = te = i+1;
            tL = 0;
        }
        else // 透明
        {
            longerL();
        }
    }

    // 顶部边界
    tL=ts=te=0;
    isCounting = false;
    for(int i=0;i<cols;i++)
    {
        if(!is_transparent(mask,0,i) || i==cols-1) // 非透明 或 已到结尾   停止计数
        {
            if(isCounting)
            {
                if(is_transparent(mask,0,i))
                {
                    tL++;
                    te++;
                }
                check_longer(tL,ts,te,Border::TOP);
            }
            isCounting = false;
            ts = te = i+1;
            tL = 0;
        }
        else // 透明
        {
            longerL();
        }
    }

    // 底部边界
    tL=ts=te=0;
    isCounting = false;
    for(int i=0;i<cols;i++)
    {
        if(!is_transparent(mask,rows-1,i) || i==cols-1) // 非透明 或 已到结尾   停止计数
        {
            if(isCounting)
            {
                if(is_transparent(mask,rows-1,i))
                {
                    tL++;
                    te++;
                }
                check_longer(tL,ts,te,Border::BOTTOM);
            }
            isCounting = false;
            ts = te = i+1;
            tL = 0;
        }
        else // 透明
        {
            longerL();
        }
    }

    return make_pair(s,e-1);
}

/**
 * 显示找到的最长的边界段
*/
void SeamCarving::show_longestborder(const Mat& img, pair<int,int> begin_end, Border direction)
{
    Mat tmp = img.clone();
    int rows = img.rows;
    int cols = img.cols;

    switch (direction)
    {
    case Border::LEFT: // 左边界
        for(int i=begin_end.first;i<=begin_end.second;i++)
            tmp.at<Vec3b>(i,0) = Vec3b(0,0,255);
        break;
    case Border::TOP: // 顶边界
        for(int i=begin_end.first;i<=begin_end.second;i++)
            tmp.at<Vec3b>(0,i) = Vec3b(0,0,255);
        break;
    case Border::RIGHT: // 右边界
        for(int i=begin_end.first;i<=begin_end.second;i++)
            tmp.at<Vec3b>(i,cols-1) = Vec3b(0,0,255);
        break;
    case Border::BOTTOM: //底边界
        for(int i=begin_end.first;i<=begin_end.second;i++)
            tmp.at<Vec3b>(rows-1,i) = Vec3b(0,0,255);
        break;
    default:
        break;
    }

    imshow("longestborder",tmp);
    waitKey(0);
}

// 根据边界方向获取 seam 方向
SeamDirection SeamCarving::get_seamdirection(Border direction)
{
    switch (direction)
    {
    case Border::BOTTOM:
    case Border::TOP:
        return SeamDirection::HORISENTAL;
        break;
    case Border::LEFT:
    case Border::RIGHT:
        return SeamDirection::VERTICAL;
        break;
    default:
        throw "Invalid direction";
        break;
    }
}

// 计算能量
Mat SeamCarving::cal_energy(const Mat& img)
{
    // 将RGB图像转化为灰度图
    Mat gray;
    cvtColor(img,gray,COLOR_BGR2GRAY);

    // 使用Sobel进行边缘检测
    Mat xgrad,ygrad,res;
    // dx=1,dy=0 计算x方向上的梯度
    Sobel(gray,xgrad,CV_8U,1,0); // 通过卷积核计算x方向上的梯度 卷积核为[[-1,0,1],[-2,0,2],[-1,0,1]] 即计算右边像素列减去左边像素列，中间的权重更大为2
    // dx=0,dy=1 计算y方向上的梯度
    Sobel(gray,ygrad,CV_8U,0,1);
    // 将xgrad和ygrad相加
    addWeighted(xgrad,0.5,ygrad,0.5,0,res);

#ifdef DEBUG_ENERGY
    // 显示图像
    imshow("xgrad",xgrad);
    imshow("ygrad",ygrad);
    imshow("res",res);
#endif

    return res;

    // 使用前向能量计算
    // 待完善
}

/**
 * 获取子图中最小的seam
 * mask中 haspixel表示为图像 missingpixel表示丢失像素
*/
int* SeamCarving::get_minimum_seam(Mat& img,Mat& mask, SeamDirection seamdirection, pair<int,int> begin_end)
{
    // 如果为水平seam则转置图像和mask转化为垂直
    if(seamdirection==SeamDirection::HORISENTAL)
    {
        transpose(img,img);
        transpose(mask,mask);
    }

    // 需要寻找seam的子图像范围
    int row_start = begin_end.first;
    int row_end = begin_end.second;
    int col_start = 0;
    int col_end = img.cols-1;

    // 获取子图
    Mat sub_img = img(Rect(col_start,row_start,col_end-col_start+1,row_end-row_start+1));
    Mat sub_mask = mask(Rect(col_start,row_start,col_end-col_start+1,row_end-row_start+1));
    int sub_rows = sub_img.rows;
    int sub_cols = sub_img.cols;

// #ifdef DEBUG_SEAM
//     // 对比图像
//     imshow("sub_img",sub_img);
//     imshow("img",img);
//     waitKey(0);
// #endif

    // 获取能量图
    Mat energy = cal_energy(sub_img);
    energy.convertTo(energy,CV_32F); // 转化为float类型，防止整数溢出
    // 处理能量图，防止seam穿过不存在的像素
    for(int i=0;i<sub_rows;i++)
        for(int j=0;j<sub_cols;j++)
            if((int)sub_mask.at<uchar>(i,j)==missingpixel)
                energy.at<float>(i,j) = INF;

    // 动态规划求解最小能量seam
    // 记录来源
    Mat backtrack = Mat::zeros(sub_rows,sub_cols,CV_32S);
    for(int i=1;i<sub_rows;i++)
    {
        for(int j=0;j<sub_cols;j++)
        {
            int idx;
            if(j==0) // 上、右上像素
            {
                idx = energy.at<float>(i-1,j) < energy.at<float>(i-1,j+1) ? j : j+1;
            }
            else if(j==sub_cols-1) // 上、左上像素
            {
                idx = energy.at<float>(i-1,j) < energy.at<float>(i-1,j-1) ? j : j-1;
            }
            else // 上、左上、右上像素
            {
                idx = energy.at<float>(i-1,j) < energy.at<float>(i-1,j-1) ? j : j-1;
                idx = energy.at<float>(i-1,idx) < energy.at<float>(i-1,j+1) ? idx : j+1;
            }
            energy.at<float>(i,j) += energy.at<float>(i-1,idx);
            backtrack.at<int>(i,j) = idx;
        }
    }

    // 找到最后一行的最小能量
    int idx = 0;
    float min_energy = energy.at<float>(sub_rows-1,0);
    for(int i=1;i<sub_cols;i++)
    {
        if(energy.at<float>(sub_rows-1,i)<min_energy)
        {
            min_energy = energy.at<float>(sub_rows-1,i);
            idx = i;
        }
    }
    // cout<<"min_energy:"<<min_energy<<endl;
    // cout<<"min_energy_idx:"<<idx<<endl;

    // 回溯找到最小能量seam
    int* seam = new int[sub_rows];
    seam[sub_rows-1] = idx;
    for(int i=sub_rows-2;i>=0;i--)
    {
        seam[i] = backtrack.at<int>(i+1,seam[i+1]);
    }

#ifdef DEBUG_SEAM
    // 显示找到的Seam
    Mat tmp = sub_img.clone();
	for(int i=0;i<sub_rows;i++)
    {
        // cout<<seam[i]<<" ";
        tmp.at<Vec3b>(i,seam[i]) = Vec3b(255,0,255);
    }
    imshow("seam",tmp);
    waitKey(0);
#endif

    // 因修改为引用传递，此处需把图片恢复原样，防止多次申请新的内存
    if(seamdirection==SeamDirection::HORISENTAL)
    {
        transpose(img,img);
        transpose(mask,mask);
    }

    return seam;
}

/**
 * 插入seam
*/
Mat SeamCarving::insert_seam(Mat& img, Mat& mask, SeamDirection seamdirection, int* seam, pair<int,int> begin_end, bool shifttoright)
{
    // 水平转化为垂直
    if(seamdirection == SeamDirection::HORISENTAL)
    {
        transpose(img,img);
        transpose(mask,mask);
    }
    
    int row_start = begin_end.first;
    int row_end = begin_end.second;
    
    for(int i=row_start;i<=row_end;i++)
    {
        int idx = i-row_start;

        if(shifttoright)
        {
            // 所有在seam右边的像素向右移动
            for(int j=img.cols-1;j>seam[idx];j--)
            {
                img.at<Vec3b>(i,j) = img.at<Vec3b>(i,j-1);
                mask.at<uchar>(i,j) = mask.at<uchar>(i,j-1);
            }
        }
        else
        {
            // 所有在seam左边的像素向左移动
            for(int j=0;j<seam[idx];j++)
            {
                img.at<Vec3b>(i,j) = img.at<Vec3b>(i,j+1);
                mask.at<uchar>(i,j) = mask.at<uchar>(i,j+1);
            }
        }

        // 平滑像素连接位置
        mask.at<uchar>(i,seam[idx]) = haspixel;
        if(seam[idx]==0)
        {
            // 最左侧
            img.at<Vec3b>(i,seam[idx]) = img.at<Vec3b>(i,seam[idx]+1);
        }
        else if(seam[idx]==img.cols-1)
        {
            // 最右侧
            img.at<Vec3b>(i,seam[idx]) = img.at<Vec3b>(i,seam[idx]-1);
        }
        else
        {
            // 中间
            img.at<Vec3b>(i,seam[idx]) = img.at<Vec3b>(i,seam[idx]-1)/2 + img.at<Vec3b>(i,seam[idx]+1)/2; // 此处计算要先/2再加，否则会出现溢出导致缝隙颜色不对
        }
    }

    if(seamdirection == SeamDirection::HORISENTAL)
    {
        transpose(img,img);
        transpose(mask,mask);
    }

    return img;
}

/**
 * 获取位移矩阵
*/
vector<vector<Coordinate>> SeamCarving::get_displacements(Mat img, Mat& mask) // 此处img非引用，调用时不影响原图
{
    if(mask.empty())
    {
        mask = getMask(img);
        cout<<"mask is empty, generate mask"<<endl;
    }
    

    vector<vector<Coordinate>> displacements; // 存储位移矩阵 记录某个点的像素来源
    vector<vector<Coordinate>> tmpdis; // 临时位移矩阵
    init_displacement(tmpdis,img.rows,img.cols);
    init_displacement(displacements,img.rows,img.cols);

    while(true)
    {
        // 不断选择最长边界
        Border direction;
        pair<int,int> begin_end = choose_longestborder(img,mask,direction);

#ifdef DEBUG_BORDER
        show_longestborder(img,begin_end,direction);
#endif

        if(begin_end.first>=begin_end.second) // 填充满了
        {
#ifdef DEBUG
            imshow("img",img);
#endif
            return displacements;
        }
        
        SeamDirection seamdirection = get_seamdirection(direction);
        bool shifttoright = direction==Border::RIGHT || direction==Border::BOTTOM;

        // 找到最小缝
        int* seam = get_minimum_seam(img,mask,seamdirection,begin_end);

        // 插入
        img = insert_seam(img,mask,seamdirection,seam,begin_end,shifttoright);

        // 更新位移矩阵
        for(int i=0;i<img.rows;i++)
        {
            for(int j=0;j<img.cols;j++)
            {
                Coordinate dis; // 记录当前像素的位移
                if(seamdirection==SeamDirection::VERTICAL && i>=begin_end.first && i<=begin_end.second)
                {
                    // 垂直缝，且在 seam 对应子图内
                    int idx = i-begin_end.first;
                    if(j>seam[idx] && shifttoright)
                    {
                        // 在缝的右侧且向右移动
                        dis.col = -1; // 要想恢复成原图像，需向左移动
                    }
                    else if(j<seam[idx] && !shifttoright)
                    {
                        // 在缝的左侧且向左移动
                        dis.col = 1; // 要想恢复成原图像，需向右移动
                    }
                }
                else if(seamdirection==SeamDirection::HORISENTAL && j>=begin_end.first && j<=begin_end.second)
                {
                    // 水平缝，且在seam 对应子图内
                    int idx = j-begin_end.first;
                    if(i>seam[idx] && shifttoright)
                    {
                        dis.row = -1;
                    }
                    else if(i<seam[idx] && !shifttoright)
                    {
                        dis.row = 1;
                    }
                }

                // 获取位移后，更新位移矩阵
                dis += Coordinate(i,j); // 当前像素加上对应的位移，即为当次位移前的坐标
                dis += tmpdis[dis.row][dis.col]; // 加上上一步存储的位移,此时即为最初始源图像的坐标
                displacements[i][j] = dis-Coordinate(i,j); // 存储位移
            }
        }

        delete[] seam;
        tmpdis = displacements;
    }
}

/**
 * 根据位移矩阵画出弯曲后图像
*/
Mat SeamCarving::get_wrapped_img(const Mat& img, const vector<vector<Coordinate>>& displacements)
{
    Mat wrapped_img = Mat::zeros(img.rows, img.cols, CV_8UC3);
    for(int i = 0; i < img.rows; i++)
    {
        for(int j = 0; j < img.cols; j++)
        {
            wrapped_img.at<Vec3b>(i, j) = img.at<Vec3b>(i+displacements[i][j].row, j+displacements[i][j].col);
        }
    }
    return wrapped_img;
}

/**
 * 放置网格
*/
vector<vector<CoordinateDouble>> SeamCarving::place_mesh(const Mat& img, const Config& config)
{
    int rows = config.rows;
    int cols = config.cols;
    int mesh_rows = config.mesh_rows;
    int mesh_cols = config.mesh_cols;
    double row_len = config.row_len;
    double col_len = config.col_len;

    vector<vector<CoordinateDouble>> mesh;
    for(int i=0;i<mesh_rows;i++)
    {
        mesh.push_back(vector<CoordinateDouble>());
        for(int j=0;j<mesh_cols;j++)
        {
            mesh[i].push_back(CoordinateDouble(i*row_len,j*col_len));
        }
    }
    return mesh;
}

/**
 * 将mesh根据位移进行变化
*/
void SeamCarving::wrap_mesh_back(vector<vector<CoordinateDouble>>& mesh, const vector<vector<Coordinate>>& displacements, const Config& config)
{
    // 网格顶点数
    int mesh_rows = config.mesh_rows;
    int mesh_cols = config.mesh_cols;

    for(int i=0;i<mesh_rows;i++)
    {
        for(int j=0;j<mesh_cols;j++)
        {
            CoordinateDouble& coord = mesh[i][j];
            const Coordinate& dis = displacements[coord.row][coord.col];
            coord.row += dis.row;
            coord.col += dis.col;
        }
    }
}


