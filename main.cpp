#include <iostream>
#include <imgRecting.h>

using namespace std;
using namespace cv;

// #define DEBUG_SAVE // 保存图片

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        cout << "Usage: " << argv[0] << " <Image_Path>" << endl;
        return -1;
    }
    // Mat image = imread(argv[1]);
    // if (image.empty())
    // {
    //     cout << "Could not read the image: " << endl;
    //     return 1;
    // }
    // imshow("Origin", image);

    imgRecting ir;
    ir.init_gl(argv[1]);
    ir.recting();
    ir.display();

#ifdef DEBUG_SAVE
    ir.save_img("./output");
#endif

    {
        // {
        //     // // 测试最小缝计算是否成功
        //     // Mat mask = getMask(image);
        //     // Border bd;
        //     // SeamCarving sc;
        //     // pair<int,int> begin_end = sc.choose_longestborder(image, mask, bd);
        //     // sc.show_longestborder(image, begin_end, bd);
        //     // SeamDirection sd = sc.get_seamdirection(bd);
        //     // sc.get_minimum_seam(image, mask, sd, begin_end);
        // }

        // // 开始计时
        // clock_t start = clock();

        // // 测试位移矩阵
        // SeamCarving sc;
        // Mat mask = getMask(image);
        // vector<vector<Coordinate>> displacements = sc.get_displacements(image, mask);

        // imshow("afterinsert", image);

        // // 查看根据位移矩阵拉伸后的图像
        // // Mat wrapped_img = sc.get_wrapped_img(image, displacements);
        // // imshow("Wrapped", wrapped_img);

        // // 画上mesh
        // Config config(image.rows, image.cols, 20, 20, 50);
        // vector<vector<CoordinateDouble>> mesh = sc.place_mesh(image, config);
        // // Mat meshon = drawMesh(image, mesh, config);
        // // imshow("Mesh", meshon);

        // // 根据位移矩阵得到形变后的mesh
        // sc.wrap_mesh_back(mesh, displacements, config);
        // // Mat wrapped_mesh = drawMesh(image, mesh, config);
        // // imshow("Wrapped_Mesh", wrapped_mesh);

        // // 查看用时
        // cout << "局部翘曲获得形变后的mesh用时: " << (clock() - start) / (double)CLOCKS_PER_SEC << "s" << endl;
        
        // // 切割线段初始化
        // GlobalWraping gw(image,mask,config);
        // vector<LineD> lineseg_flatten; // 所有边
        // vector<pair<int,double>> id_theta; // 每条边的id以及角度
        // vector<double> rotate_theta; // 每条边旋转的角度
        // vector<vector<vector<LineD>>> line_seg_mesh = gw.init_line_seg(mesh, lineseg_flatten, id_theta, rotate_theta);

        // vector<pair<MatrixXd, MatrixXd>> BilinearMat_Vec; // 需要更新的线段的双线性插值矩阵
        // int n_lines;
        // SparseMatrix<double,RowMajor> line_mat = gw.get_line_mat(mesh, line_seg_mesh, rotate_theta, BilinearMat_Vec, n_lines);

        // waitKey(0);
    }    
    return 0;
}