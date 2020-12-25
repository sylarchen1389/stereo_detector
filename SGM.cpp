#include"SGM.h"
#include"libsgm.h"

SGM::SGM(SGMOption option)
{
    option_ = option;    

	// ASSERT_MSG(rawLeft.type() == CV_8U || rawLeft.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
	ASSERT_MSG(option.disp_size == 64 || option.disp_size == 128 || option.disp_size == 256, "disparity size must be 64, 128 or 256.");
	ASSERT_MSG(option.num_paths == 4 || option.num_paths == 8, "number of scanlines must be 4 or 8.");

	const sgm::PathType path_type = option.num_paths == 8 ? sgm::PathType::SCAN_8PATH : sgm::PathType::SCAN_4PATH;
	// const int input_depth = rawLeft.type() == CV_8U ? 8 : 16;;
	const int input_depth  = 8;
	const int output_depth = 16;

    const sgm::StereoSGM::Parameters param( option_.p1, option_.p2_int, option_.uniqueness,
        option_.subpixel,path_type, option_.min_disp, option_.LR_max_diff);
    
    std::cout <<"sgm param: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<std::endl;
    std::cout<<"imageWidth: "<<option_.imageWidth<<std::endl
    <<"imageHeight: "<<option_.imageHeight<<std::endl
    <<"disp_size: "<<option_.disp_size<<std::endl
    <<"input_depth: "<<input_depth<<std::endl
    <<"output_depth: "<<output_depth<<std::endl
    <<"p1: "<<option_.p1<<std::endl
    <<"p2_int: "<<option_.p2_int<<std::endl
    <<"uniqueness: "<<option_.uniqueness<<std::endl
    <<"subpixel: "<<option_.subpixel<<std::endl
    <<"min_disp: "<<option_.min_disp<<std::endl
    <<"LR_max_diff: "<<option_.LR_max_diff<<std::endl;
    std::cout <<"sgm param: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;

    ssgm_.reset(new sgm::StereoSGM( option_.imageWidth, option_.imageHeight,
        option_.disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_HOST2HOST, param));

}

SGM::~SGM()
{
}


void SGM::getXYZ(cv::Mat& grayImageL,cv::Mat& grayImageR,cv::Mat& xyzOut,cv::Mat& disparityOut){
    ASSERT_MSG(grayImageL.size() == grayImageR.size() && 
        grayImageL.type() == grayImageR.type(), "input images must be same size and type.");
    ASSERT_MSG(grayImageL.type() == CV_8U || grayImageL.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
	
    grayImageL.copyTo(grayImageL_);
    grayImageR.copyTo(grayImageR_);

    // try{
    //     cv::Mat showL,showR;
    //     cv::resize(grayImageL_,showL,cv::Size(1920,768));
    //     cv::resize(grayImageR_,showR,cv::Size(1920,768));
    //     cv::imshow("left",showL);
    //     cv::imshow("right",showR);
    // }catch(...){
    //     std::cout<<"show gray image fail."<<std::endl;
    // }
    cv::waitKey(30);

    std::cout<<"[INFO] sgm recevie"<<grayImageR_.cols<<","<<grayImageR_.rows<<std::endl;
    /*计算视差*/
    cv::Mat disparity(grayImageL.size(), CV_16S);
    ssgm_->execute(grayImageL_.data, grayImageR_.data, disparity.data);
    std::cout<<"[INFO]disparity: "<<disparity.cols<<","<<disparity.rows<<std::endl;

    cv::Mat disparity_8u, disparity_color;
    disparity.copyTo(disparityOut);
    cv::Mat xyz;
    compute3d(disparity,xyz, 1800.57748439764,1/0.1256199817826959,1313.028991699219,428.343528747559);
    // compute3d(disparity,xyz, 1333.761099554,8.01614898702783,972.614067925,317.291502776);
    xyz.copyTo(xyzOut);

    cv::Mat Q = (cv::Mat_<double>(4,4) << 
        1, 0, 0, -1313.028991699219,
        0, 1, 0, -428.343528747559,
        0, 0, 0, 1800.57748439764,
        0, 0, 0.1256199817826959, -0);

    /*计算3d坐标*/ 
	// cv::Mat disp;
    // cv::Mat XYZ;
	// disparity.convertTo(disp,CV_32F,1.0);    
	// reprojectImageTo3D(disp, XYZ, Q, true,CV_32F);
    // XYZ.copyTo(xyzOut);

    

    /*show image[for test]*/
	disparity.convertTo(disparity_8u, CV_8U, 255/64);
	cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);
    cv::Mat mask = disparity == ssgm_->get_invalid_disparity();
	disparity_8u.setTo(0, mask);
	disparity_color.setTo(cv::Scalar(0, 0, 0), mask);
    disparity_color.copyTo(disparityOut);

    // try{
	// 	cv::resize(disparity_8u,disparity_8u,cv::Size(1920,768));
	// 	cv::resize(disparity_color,disparity_color,cv::Size(1920,768));
	// 	cv::imshow("disparity8U",disparity_8u);
	// 	cv::imshow("disparityColor",disparity_color);
	// }catch(cv::Exception e){
	// 	std::cout<<e.msg<<std::endl;
	// }
    // cv::waitKey(30);
    
}



void SGM::compute3d(cv::Mat& disparity,cv::Mat& xyz,
    float force,float baseline,float cxl,float cyl)
{
    cv::Mat XYZ(disparity.size(),CV_32FC3);
    for(int row = 0 ; row<disparity.rows;row++){
        for(int col =0;col <disparity.cols;col++){
            int16_t d = disparity.ptr<int16_t>(row)[col];
            // float d_f = (float)(d + 1323.68935404431 -1313.028991699219);
            // int d = d16 + 1285 - 1299 ;

            if(d <= 0 || d>=127){
                // std::cout<<row<<","<<col<<": "<<d<<std::endl;
                cv::Vec3f point;
                point[0] = point[1] = point[2] = 160000;
                XYZ.at<cv::Vec3f>(row,col) = point;
            }

            // d += 1;

            cv::Vec3f point;
            point[2] = baseline*force/d;
            point[0] = point[2]*(col + 1 - cxl)/force;
            point[1] = point[2]*(row + 1 - cyl )/force;
            XYZ.at<cv::Vec3f>(row,col) = point;
        }
    }
    XYZ.copyTo(xyz);
}