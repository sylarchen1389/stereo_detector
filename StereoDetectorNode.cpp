#include"StereoDetectorNode.h"



StereoDetectorNode::StereoDetectorNode(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private)
    : nh_(nh), nh_private_(nh_private), imageTransport_(nh)
{

    string cameraTopicLeft;
    string cameraTopicRight;
    string detectionsImageTopic;
    string disparityImageTopic;
    string bboxTopic;
    string carType;
    double timeout;

#pragma region launchParam
    /*ros*/
    nh_private_.param("camera_topic_left",cameraTopicLeft,std::string("/camera/leftRaw"));
    nh_private_.param("camera_topic_right",cameraTopicRight,std::string("/camera/rightRaw"));
    nh_private_.param("detections_topic",detectionsImageTopic,std::string("/stereo_detector/detectionsImage"));
    nh_private_.param("disparity_topic",disparityImageTopic,std::string("/stereo_detector/disparityImage"));
    nh_private_.param("bboxes_topic",bboxTopic,std::string("/stereo_detector/bboxes"));
    /*tensorrt*/
    nh_private_.param("onnx_path",trtOption_.onnxFile,string("yolov3.onnx"));
    nh_private_.param("trt_path",trtOption_.trtFile,string("yolov3.trt"));
    nh_private_.param("calib_path",trtOption_.calibFile,string("yolo.txt"));
    nh_private_.param("timeout",timeout,2.0);
    nh_private_.param("yolo_width",trtOption_.yoloW,800);
    nh_private_.param("yolo_height",trtOption_.yoloH,800);
    nh_private_.param("yolo_classes",trtOption_.yoloClasses,80);
    nh_private_.param("yolo_detection_threshold",trtOption_.yoloThresh,0.998);
    nh_private_.param("yolo_nms_threshold",trtOption_.yoloNms,0.25);
    nh_private_.param("car_type",carType,string("hrt20d"));
    nh_private_.param("use_int8",trtOption_.useInt8,false);
    //nh_private.param("box_min_size_ratio",boxMinSizeRatio_,0.012);
    /*sgm*/
    nh_private_.param("image_width",sgmOption_.imageWidth,2952);
    nh_private_.param("image_height",sgmOption_.imageHeight,1038);
    nh_private_.param("disp_size",sgmOption_.disp_size,128);
    nh_private_.param("p1",sgmOption_.p1,10);
    nh_private_.param("p2_int",sgmOption_.p2_int,220);
    nh_private_.param("uniqueness",sgmOption_.uniqueness,0.99f);
    nh_private_.param("subpixel",sgmOption_.subpixel,false);
    nh_private_.param("num_paths",sgmOption_.num_paths,8);
    nh_private_.param("min_disp",sgmOption_.min_disp,0);
    nh_private_.param("lr_max_diff",sgmOption_.LR_max_diff,1);
    nh_private_.param("baseline",sgmOption_.baseline,8.01614898702783f);
    nh_private_.param("force",sgmOption_.force,1802.65855619198f);
    nh_private_.param("cxl",sgmOption_.cxl,1299);
    nh_private_.param("cxr",sgmOption_.cxr,1285);
    nh_private_.param("cyl",sgmOption_.cyl,424);
    nh_private_.param("cyl",sgmOption_.cyr,453);


    trtOption_.onnxFile = ros::package::getPath("stereo_detector")+"/"+trtOption_.onnxFile;
    trtOption_.trtFile = ros::package::getPath("stereo_detector")+"/"+trtOption_.trtFile;
    trtOption_.calibFile = ros::package::getPath("stereo_detector")+"/"+trtOption_.calibFile;
#pragma endregion
#pragma region camParam
    camOption_.cameraMatrixL = (cv::Mat_<double>(3, 3) <<
        1799.98852787378,0,1323.68935404431,
        0,1799.10577181371,1023.97977890216,
        0,0,1);
    camOption_.distCoeffL = (cv::Mat_<double>(5, 1) << 
        -0.147055131065462,0.150449210882043,-0.00133156154198220,0.00211672773185521,-0.0421072966690359);

    camOption_.cameraMatrixR = (cv::Mat_<double>(3, 3) << 
        1800.14279727335,0,1307.88868461770,
        0,1799.39390950049,1040.29964105710,
        0,0,1);
    camOption_.distCoeffR = (cv::Mat_<double>(5, 1) << 
        -0.148803965819541,0.152691806471188,-0.00124043605469856,0.00278208589264770,-0.0448779491567325);

    camOption_.T = (cv::Mat_<double>(3, 1) << -7.96016216343397,0.0646698136337603,-0.0383012834848667);//T平移向量
    //Mat rec = (Mat_<double>(3, 1) << -0.00306, -0.03207, 0.00206);//rec旋转向量
    camOption_.R = (cv::Mat_<double>(3,3)<<
        0.999920586961890,-0.0123262085575864,-0.00262380494413116,
        0.0123436584137887,0.999901087092108,0.00674167108252923,
        0.00254044617216446,-0.00677352305792141,0.999973832416944);//R 旋转矩阵
    camOption_.imageHeight = 1038;
    camOption_.imageWidth = 2592;

    double cxl = 1323.68935404431;
    double cxr = 1307.88868461770;

    float baseline = 7.96016216343397; 
    float force = 1799.98852787378;

    // t_mat.at<float>(0, 0) = 1;
	// t_mat.at<float>(0, 2) = 0; //水平平移量
	// t_mat.at<float>(1, 1) = 1;
	// t_mat.at<float>(1, 2) = 0; //竖直平移量
#pragma endregion


    imageSubscriber1_.subscribe(imageTransport_, cameraTopicLeft,3);
    imageSubscriber2_.subscribe(imageTransport_, cameraTopicRight,3);

    sync_.reset(new message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>>(message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>(5), imageSubscriber1_, imageSubscriber2_));
    sync_->registerCallback(boost::bind(&StereoDetectorNode::cameraCallback, this, _1, _2));

    boundingBoxesPublisher_ = nh_.advertise<stereo_detector::BoundingBoxes>(bboxTopic, 2, false);
    detectionImagePublisher_ = imageTransport_.advertise(detectionsImageTopic, 2);
    disparityImagePublisher_ = imageTransport_.advertise(disparityImageTopic,2);

    images_.reset(new ThreadSafeImage());
    bboxes_.reset(new ThreadSafeBoundingBoxes());

    tensorrtThread_.reset(new boost::thread(boost::bind(&StereoDetectorNode::computeBBox, this)));
    sgmThread_.reset(new boost::thread(boost::bind(&StereoDetectorNode::computeXYZ, this)));
}

StereoDetectorNode::~StereoDetectorNode()
{
    if(tensorrtThread_){
        tensorrtThread_->interrupt();
        tensorrtThread_->join();
    }

    if(sgmThread_){
        sgmThread_->interrupt();
        sgmThread_->join();
    }

    cv::destroyAllWindows();
}

void StereoDetectorNode::rectify(
    cv::Mat& left,
    cv::Mat& right)
{

    /*立体矫正*/
    cv::Size originSize(2592,2048);
    cv::stereoRectify(camOption_.cameraMatrixL, camOption_.distCoeffL, camOption_.cameraMatrixR,camOption_.distCoeffR, 
        originSize, camOption_.R, camOption_.T, camOption_.Rl, camOption_.Rr, camOption_.Pl, camOption_.Pr, camOption_.Q, 
        cv::CALIB_ZERO_DISPARITY,0, originSize);
    initUndistortRectifyMap(camOption_.cameraMatrixL, camOption_.distCoeffL, camOption_.Rl, camOption_.Pl, 
        originSize, CV_32FC1, camOption_.mapLx, camOption_.mapLy);
    initUndistortRectifyMap(camOption_.cameraMatrixR, camOption_.distCoeffR, camOption_.Rr, camOption_.Pr, 
        originSize, CV_32FC1, camOption_.mapRx, camOption_.mapRy);
    
    camOption_.Q.copyTo(Q_);

    // std::cout<<"Q="<<endl<<camOption_.Q<<endl;
    
    cv::Mat rectLeft,rectRight;
    remap(left, rectLeft, camOption_.mapLx, camOption_.mapLy, cv::INTER_LINEAR);
    remap(right, rectRight, camOption_.mapRx, camOption_.mapRy, cv::INTER_LINEAR);
   
    // try{
    //     cv::Mat showL,showR;
    //     cv::resize(rectLeft,showL,cv::Size(2592/4,2048/4));
    //     cv::resize(rectRight,showR,cv::Size(2592/4,2048/4));
    //     cv::imshow("orginL",showL);
    //     cv::waitKey(10);
    //     cv::imshow("orginR",showR);
    // }catch(...){
// 
    // }

    // ########
    rectLeft =  rectLeft(cv::Rect(0, 600, 2592, 1038));
	rectRight =  rectRight(cv::Rect(0, 600, 2592, 1038));
    // ########

    // cv::resize(rectLeft,rectLeft,cv::Size(sgmOption_.imageWidth,sgmOption_.imageHeight));
    // cv::resize(rectRight,rectRight,cv::Size(sgmOption_.imageWidth,sgmOption_.imageHeight));

    rectLeft.copyTo(left);
    rectRight.copyTo(right);
    
}


void StereoDetectorNode::cameraCallback(
    const sensor_msgs::ImageConstPtr& msg1, 
    const sensor_msgs::ImageConstPtr& msg2)
{
    cv_bridge::CvImagePtr RectifyImageL;
    cv_bridge::CvImagePtr RectifyImageR;

    cout<<"[INFO] left cam timestamp: "<< msg1->header.stamp<<endl;
    cout<<"[INFO] right cam timestamp: "<< msg2->header.stamp<<endl;

    try {
        RectifyImageL = cv_bridge::toCvCopy(msg1, sensor_msgs::image_encodings::BGR8);
        RectifyImageR = cv_bridge::toCvCopy(msg2, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    //  if raw frame
    cv::Mat rawLeft,rawRight;
    RectifyImageL->image.copyTo(rawLeft);
    RectifyImageR->image.copyTo(rawRight);
    rectify(rawLeft,rawRight);
    
    
    auto t_start = std::chrono::high_resolution_clock::now();
    // images_->wirteRectifyImageL(RectifyImageL->image);
    images_->wirteRectifyImageL(rawLeft);
    
    cv::Mat grayImageL,grayImageR;
    // cv::cvtColor(RectifyImageL->image,grayImageL,CV_BGR2GRAY);
    // cv::cvtColor(RectifyImageR->image,grayImageR,CV_BGR2GRAY);
    cv::cvtColor(rawLeft,grayImageL,CV_BGR2GRAY);
    cv::cvtColor(rawRight,grayImageR,CV_BGR2GRAY);
    std::cout<<grayImageL.cols<<","<<grayImageL.rows<<endl;
    images_->writeGrayImage(grayImageL,grayImageR);  

    cv::Mat xyz;
    images_->getXYZ(xyz);

    if(xyz.empty()||xyz.rows == 0||xyz.cols == 0){
        cerr<<"xyz is empty!"<<endl;
        return;
    }

    std::vector<BoundingBox> detections;
    bboxes_->getBBoxes(detections);
    cout<<"[INFO]bboxes size:"<<detections.size()<<endl;

    /*计算bbox的三维坐标*/
    int x,y;
    int count;
    float worldX,worldY,worldZ;
    int pad = 3;
    stereo_detector::BoundingBoxes boxes;
    for(BoundingBox box: detections){
        x = box.xmin + (box.xmax - box.xmin)/2;
        y = box.ymin + (box.ymax - box.ymin)/2;
        count = 0;
        for(int i = y-pad;i<y+pad;i++){
            for(int j= x-pad;j<x+pad;j++){
                cv::Vec3f point = xyz.at<cv::Vec3f>(i, j);
                if(fabs(point[2]) == 160000||fabs(point[1])== 160000||fabs(point[0])== 160000||
                    point[2] <= 0||fabs(point[1])== 0 ||fabs(point[0])== 0){
                    continue;
                }else{
                    worldX += point[0];
                    worldY += point[1];
                    worldZ += point[2];              
                    count ++;
                }
            }
        }
        if(count == 0){
            cerr<< "empty point !"<<endl;
            continue;
        }
        box.worldX = worldX = worldX/count; 
        box.worldY = worldY = worldY/count; 
        box.worldZ = worldZ = worldZ/count; 

        cout<<"[INFO]World Poistion: "
            << "worldX: "<<worldX<<","
            <<"worldY: "<<worldY<<","
            <<"worldZ: "<<worldZ<<endl;

        stereo_detector::BoundingBox boundingBox;
        boundingBox.probability = box.probability;
        boundingBox.xmin = box.xmin;
        boundingBox.ymin = box.ymin;
        boundingBox.xmax = box.xmax;
        boundingBox.ymax = box.ymax;
        boundingBox.Class = stoi(box.Class);
        boundingBox.worldX = box.worldX;
        boundingBox.worldY = box.worldY;
        boundingBox.worldZ = box.worldZ; 
        boxes.bounding_boxes.push_back(boundingBox);
        
        // 绘图
        if(drawDetections_){
            // cv::rectangle(RectifyImageL->image,cv::Point(box.xmin,box.ymin),
            // cv::Point(box.xmax,box.ymax),(0,0,255));
            // cv::putText(RectifyImageL->image,box.Class,cv::Point(box.xmax,box.ymin),cv::FONT_HERSHEY_SIMPLEX   ,0.4,(255,0,0));
            // cv::putText(RectifyImageL->image, std::to_string(box.probability),cv::Point(box.xmax,box.ymin+15),cv::FONT_HERSHEY_SIMPLEX,0.4,(255,0,0));
            // cv::putText(RectifyImageL->image, "x: "+std::to_string(box.worldX),cv::Point(box.xmax,box.ymin+30),cv::FONT_HERSHEY_SIMPLEX,0.4,(255,0,0));
            // cv::putText(RectifyImageL->image, "y: "+std::to_string(box.worldY),cv::Point(box.xmax,box.ymin+45),cv::FONT_HERSHEY_SIMPLEX,0.4,(255,0,0));
            // cv::putText(RectifyImageL->image, "z: "+std::to_string(box.worldZ),cv::Point(box.xmax,box.ymin+60),cv::FONT_HERSHEY_SIMPLEX,0.4,(255,0,0));
            cv::rectangle(rawLeft,cv::Point(box.xmin,box.ymin),
            cv::Point(box.xmax,box.ymax),(0,0,255));
            cv::putText(rawLeft,box.Class,cv::Point(box.xmax,box.ymin),cv::FONT_HERSHEY_SIMPLEX   ,0.4,(255,0,0));
            cv::putText(rawLeft, std::to_string(box.probability),cv::Point(box.xmax,box.ymin+15),cv::FONT_HERSHEY_SIMPLEX,0.4,(255,0,0));
            cv::putText(rawLeft, "x: "+std::to_string(box.worldX),cv::Point(box.xmax,box.ymin+30),cv::FONT_HERSHEY_SIMPLEX,0.4,(255,0,0));
            cv::putText(rawLeft, "y: "+std::to_string(box.worldY),cv::Point(box.xmax,box.ymin+45),cv::FONT_HERSHEY_SIMPLEX,0.4,(255,0,0));
            cv::putText(rawLeft, "z: "+std::to_string(box.worldZ),cv::Point(box.xmax,box.ymin+60),cv::FONT_HERSHEY_SIMPLEX,0.4,(255,0,0));
        
        }
    }

    boxes.header = RectifyImageL->header;
    boundingBoxesPublisher_.publish(boxes);

    if(drawDetections_)
        detectionImagePublisher_.publish(RectifyImageL->toImageMsg());

    try{
        cv::resize(rawLeft,rawLeft,cv::Size(1920,768));
        cv::imshow("drawDetection",rawLeft);
        cv::waitKey(3);
    }catch(...){}
    

    auto t_end = std::chrono::high_resolution_clock::now();
    auto total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "[INFO] TOTAL CALLBACK TIME" << total << " ms." << std::endl;


}


/*深度计算线程*/
void StereoDetectorNode::computeXYZ(){

    SGM sgm(sgmOption_);
    std::cout<<"[INFO]create sgm done"<<std::endl;
    // sleep(1000);
    while (true)
    {
        Q_.copyTo(sgm.Q_ );
        cv::Mat grayImageL,grayImageR;
        cv::Mat XYZ,disparity;
        images_->getGrayImage(grayImageL,grayImageR);
        sgm.getXYZ(grayImageL,grayImageR,XYZ,disparity);
        cv::resize(disparity,disparity,cv::Size(1920,768));
        cv::imshow("disparity",disparity);
        cv::waitKey(3);
        images_->writeXYZ(XYZ);
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(),"bgr8",disparity).toImageMsg();
        disparityImagePublisher_.publish(msg);
    }
    
}

/*目标检测线程*/
void StereoDetectorNode::computeBBox(){
    
    TensorRT tensorrt(trtOption_);
    // tensorrt.reset(new TensorRT(option));
    std::cout<<"[INFO]create tensorrt done"<<std::endl;
    while (true)
    {
        cv::Mat rectifyImageL;
        images_->getRectifyImageL(rectifyImageL);
        std::vector<BoundingBox> boxes;
        tensorrt.detect(rectifyImageL,boxes);
        bboxes_->wirteBBoxws(boxes);
    }
}