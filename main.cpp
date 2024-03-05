//导入opencv,yolop库
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "apps/yolop/yolop.hpp"
#include <cmath>

//串口发送库
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>
#include <iconv.h>
#include <string>
#include <iostream>

using namespace std;



/*
//拍摄一段视频进行保存

//CSI 摄像头检测
//CSI 摄像头管道处理
std::string gstreamer_pipeline(int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)I420 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

int main() {
    int capture_width = 1280;
    int capture_height = 720;
    int display_width = 1280;
    int display_height = 720;
    int framerate = 30;
    int flip_method = 0;

    std::string pipeline = gstreamer_pipeline(capture_width,
                                              capture_height,
                                              display_width,
                                              display_height,
                                              framerate,
                                              flip_method);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cout << "无法打开摄像头" << std::endl;
        return -1;
    }

    int width = 640;
    int height = 480;

    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

    cv::VideoWriter video("/home/rory/Projects/Linfer/workspace/videos/output.mp4", cv::VideoWriter::fourcc('H', '2', '6', '4'), framerate, cv::Size(width, height));
    // 设置保存视频的名称为output.mp4，视频编解码器为H.264，帧率为30fps，大小为设置的宽度和高度
    if (!video.isOpened()) {
        std::cout << "无法创建视频文件" << std::endl;
        return -1;
    }

    while (true) {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cout << "无法捕获帧" << std::endl;
            break;
        }

        video.write(frame);

        cv::imshow("CSI Camera", frame);

        char key = cv::waitKey(1);
        if (key == 27)
            break;
    }

    cap.release();
    video.release();

    cv::destroyAllWindows();

    return 0;
}
*/








//用直线角度和与下方横坐标的交点判断（第一版）
/*
// 定义直线模型结构体
struct LineModel {
    double slope;
    double intercept;
};

// 计算两点间直线斜率和截距
LineModel computeLineModel(const cv::Point& pt1, const cv::Point& pt2) {
    LineModel model;
    if (pt1.x != pt2.x) {
        model.slope = static_cast<double>(pt2.y - pt1.y) / (pt2.x - pt1.x);
        model.intercept = pt1.y - model.slope * pt1.x;
    } else {
        // 处理斜率为无穷大的情况
        model.slope = std::numeric_limits<double>::infinity();
        model.intercept = std::numeric_limits<double>::quiet_NaN();
    }
    return model;
}

// 使用RANSAC算法拟合直线
LineModel RANSACLineFitting(const std::vector<cv::Point>& points, int iterations, double threshold) {
    const int numPoints = points.size();
    if (numPoints < 2) {
        throw std::invalid_argument("Insufficient points for fitting");
    }

    LineModel bestModel;
    int bestInliers = 0;

    for (int i = 0; i < iterations; ++i) {
        // 随机选择两个点
        int idx1 = std::rand() % numPoints;
        int idx2 = std::rand() % numPoints;
        while (idx1 == idx2) {
            idx2 = std::rand() % numPoints;
        }

        const cv::Point& pt1 = points[idx1];
        const cv::Point& pt2 = points[idx2];

        // 计算当前模型
        LineModel currentModel = computeLineModel(pt1, pt2);

        // 统计符合阈值内的内点数
        int inliers = 0;
        for (const auto& pt : points) {
            double distance = std::abs(currentModel.slope * pt.x - pt.y + currentModel.intercept) /
                              std::sqrt(currentModel.slope * currentModel.slope + 1);
            if (distance < threshold) {
                inliers++;
            }
        }

        // 更新最佳模型
        if (inliers > bestInliers) {
            bestInliers = inliers;
            bestModel = currentModel;
        }
    }

    return bestModel;
}

void drawLine(cv::Mat& image, const LineModel& lineModel) {
    // 直线的两个端点
    cv::Point pt1(0, static_cast<int>(lineModel.intercept));
    cv::Point pt2(image.cols, static_cast<int>(lineModel.slope * image.cols + lineModel.intercept));

    // 在图像上绘制直线
    cv::line(image, pt1, pt2, cv::Scalar(255, 0, 0), 2);
}

void drawPoints(cv::Mat& image, const std::vector<cv::Point>& points) {
    for (const auto& point : points) {
        cv::circle(image, point, 3, cv::Scalar(255, 0, 0), cv::FILLED);
    }
}

// 计算角度的函数
double calculateAngle(const LineModel& lineModel) {
    double angle = atan(lineModel.slope); // 使用斜率计算弧度角

    // 将角度转换为度数
    angle = angle * 180.0 / CV_PI;
    //printf("angle:%f\n",angle);
    // 调整角度，考虑参考方向
    // 如果斜率为负，则直线向右倾斜
    // 如果斜率为正，则直线向左倾斜
    // 为了统一，以直线的上半部分为参考方向，以正右为0度，正左为180度
    if (lineModel.slope < 0) {
        angle =abs(angle);
    } else {
        angle = 180.0 - angle;
    }

    return angle;
}

// 计算直线与图像底部的交点
int calculateBottomIntersectionX(const LineModel& lineModel, int imageHeight) {
    // 对于底部，y 的值是图像的高度
    int y = imageHeight;

    // 使用直线方程求解 x
    double x = 0.0;
    if (lineModel.slope != std::numeric_limits<double>::infinity()) {
        // 计算交点的 x 坐标（斜率不是无穷大的情况）
        x = (y - lineModel.intercept) / lineModel.slope;
    } else {
        // 处理斜率为无穷大的情况（直线与 x 轴平行）
        x = lineModel.intercept; // 直线与 x 轴的交点即为截距
    }

    return static_cast<int>(x); // 返回交点横坐标
}


//CSI 摄像头检测
//CSI 摄像头管道处理
std::string gstreamer_pipeline(int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)I420 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

int main() {
    // 创建yolopv2检测器
    auto detector = YoloP::create_detector("/home/rory/Projects/Linfer/workspace/yolopv2-480x640.trt", YoloP::Type::V2, 0, 0.4, 0.5);
    
    // 预热处理20张图片以免启动时较慢影响判断
    auto image = cv::imread("/home/rory/Projects/Linfer/workspace/imgs/1.jpg");
    YoloP::PTMM res;
    for(int i = 0; i < 20; ++i)
        res = detector->detect(image);
    
    //USB摄像头
    //cv::VideoCapture cap(0);
    
    //CSI摄像头
    int capture_width = 1280 ;
    int capture_height = 720 ;
    int display_width = 1280 ;
    int display_height = 720 ;
    int framerate = 30 ;
    int flip_method = 0 ;

    std::string pipeline = gstreamer_pipeline(capture_width,
	capture_height,
	display_width,
	display_height,
	framerate,
	flip_method);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n";
 
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    //视频检测
    //cv::VideoCapture cap("/home/rory/Projects/Linfer/workspace/videos/video2.mp4");
    if (cap.isOpened()) {
        cv::namedWindow("CSI Camera", cv::WINDOW_AUTOSIZE);
        while (cv::getWindowProperty("CSI Camera", 0) >= 0) {
            cv::Mat img;
            cap >> img;

            //缩放为宽：640，高：480
            int width = 640;
            int height = 480;
            cv::resize(img,img, cv::Size(width, height));

            // 开始检测
            auto start = std::chrono::steady_clock::now();

            auto result = detector->detect(img); // 使用处理后的图像进行检测
            // 获取帧率
            std::chrono::duration<double> during = std::chrono::steady_clock::now() - start;
            double use_time = 1000.0 * during.count();
            printf("FPS: %.2f\n", 1000 / use_time);

            // 0对应车辆框，1对应驾驶区域，2对应车道线
            YoloP::BoxArray& boxes = get<0>(result);
            cv::Mat& drive_mask = get<1>(result);
            cv::Mat& lane_mask = get<2>(result);

    //转化为灰度图，将通道转化为1，不然后面图像处理有问题
    //cv::cvtColor(lane_mask, lane_mask, cv::COLOR_BGR2GRAY);
    //二值化
    int threshold_value = 128; // 设定阈值
    int max_value = 255; // 最大值（白色）
    cv::threshold(lane_mask, lane_mask, threshold_value, max_value, cv::THRESH_BINARY);


    //printf("width:%d,height:%d\n", lane_mask.cols, lane_mask.rows);
    //cols是宽，rows是高

     //腐蚀
     cv::Mat kernel = cv::Mat::ones(3, 5, CV_8U);
     cv::erode(lane_mask, lane_mask, kernel);

     int middle = width/2;
     int start_ = 1;
     int end_ = height;
     int step = 5;

    std::vector<cv::Point> points;

     for (int i = start_; i < end_; i += step)
     {
         
        //   //直接查找
        //   int left_point = -1;
        //   for (int j = middle; j > 0; --j) {
        //       if (lane_mask.at<uchar>(i, j)) {
        //           left_point = j;
        //           printf("left_point:%d\n",left_point);
        //           break;
        //       }
        //   }

        //   int right_point = -1;
        //   for (int j = middle; j < width; ++j) {
        //       if (lane_mask.at<uchar>(i, j)) {
        //           right_point = j;
        //           printf("right_point:%d\n",right_point);
        //           break;
        //       }
        //   }

         //二分查找
         int left_point = -1;
         int right_point = -1;

         // 从中点向左搜索最近的点
         int left = middle;
         int right = middle - 1;
         while (left >= 0 && right_point == -1) {
             if (lane_mask.at<uchar>(i, left)) {
                 left_point = left;
                 break;
             }
             if (lane_mask.at<uchar>(i, right)) {
                 right_point = right;
                 break;
             }
             left--;
             right--;
         }

         // 从中点向右搜索最近的点
         left = middle + 1;
         right = middle;
         while (right < width && left_point == -1) {
             if (lane_mask.at<uchar>(i, right)) {
                 right_point = right;
                 break;
             }
             if (lane_mask.at<uchar>(i, left)) {
                 left_point = left;
                 break;
             }
             left++;
             right++;
         }

         // 如果找到了左右两边的点，计算中间位置的点
         if (left_point != -1 && right_point != -1) {
             // 计算中间位置的点
             cv::Point point;
             point.x = (left_point + right_point) / 2;
             point.y = i;
             points.push_back(point); // 将点添加到列表中
         }
     }

    if(points.size()>=8){
    
     //使用RANSAC算法拟合直线
     //设置RANSAC参数
    int iterations = 5; // 迭代次数
    double threshold = 5.0; // 内点阈值

    // 进行RANSAC直线拟合
    LineModel bestLine = RANSACLineFitting(points, iterations, threshold);

    drawPoints(lane_mask, points);
    // 绘制拟合直线到图像
    drawLine(lane_mask, bestLine);

    // 计算直线与水平轴的夹角
    double lineAngle = calculateAngle(bestLine);
    std::cout << "直线与右向水平轴的夹角：" << lineAngle << " 度" << std::endl;
    // 获得直线与图像底部交点的横坐标
    int bottomIntersectionX = calculateBottomIntersectionX(bestLine, height);
    std::cout << "直线与图像底部的交点横坐标：" << bottomIntersectionX << std::endl;
    //横坐标与图像宽度的比例
    double ratio = static_cast<double>(bottomIntersectionX) / width;
    std::cout << "横坐标与图像宽度的比例：" << ratio << std::endl;
    }
    
    // 将可驾驶区域像素设置为绿色
    img.setTo(cv::Scalar(0, 255, 0), drive_mask);

    //将白线区域像素设置成蓝色
    img.setTo(cv::Scalar(255, 0, 0), lane_mask);

    cv::imshow("CSI Camera", img);

    int keyCode = cv::waitKey(1) & 0xFF;
    if (keyCode == 27)
    {  
        // ESC键退出
        break;
    }
        }

    cap.release();
    cv::destroyAllWindows();
    } else {
        std::cout << "打开摄像头失败" << std::endl;
    }

    return 0;
}
*/





//求出自己行驶车道旁边的两条线缺口的两个点的中点（第二版）

//CSI 摄像头检测
//CSI 摄像头管道处理
std::string gstreamer_pipeline(int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)I420 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

// 找到最顶部的点
std::vector<cv::Point> findTopmostPoints(const std::vector<std::vector<cv::Point>>& contours) {
    std::vector<cv::Point> topmostPoints;

    for (const auto& contour : contours) {
        auto topmostPoint = std::min_element(contour.begin(), contour.end(),
                                             [](const cv::Point& p1, const cv::Point& p2) {
                                                 return p1.y < p2.y;
                                             }
        );
        topmostPoints.push_back(*topmostPoint);
    }

    return topmostPoints;
}

//求出轮廓中心点
std::vector<cv::Point2f> calculateLaneCenters(const std::vector<std::vector<cv::Point>>& laneContours) {
    std::vector<cv::Point2f> laneCenters;

    for (const auto& laneContour : laneContours) {
        cv::Moments moments = cv::moments(laneContour);
        if (moments.m00 != 0) {
            cv::Point2f center(static_cast<float>(moments.m10 / moments.m00),
                               static_cast<float>(moments.m01 / moments.m00));
            laneCenters.push_back(center);
        }
    }

    return laneCenters;
}

//枚举变量判断是否全在左侧或者右侧
enum LanePosition {
    ALL_LEFT,
    ALL_RIGHT,
    MIXED
};

//判断是否全在左侧或者右侧
LanePosition checkLanePositions(const std::vector<cv::Point2f>& laneCenters, int imageCenterX) {
    bool allLeft = true;
    bool allRight = true;

    for (const auto& center : laneCenters) {
        if (center.x > imageCenterX) {
            allLeft = false;
        } else {
            allRight = false;
        }
    }

    if (allLeft) {
        return LanePosition::ALL_LEFT;
    } else if (allRight) {
        return LanePosition::ALL_RIGHT;
    } else {
        return LanePosition::MIXED;
    }
}

//筛选出纵坐标差不多的点
std::vector<cv::Point> filterPointsByVertical(std::vector<cv::Point>& points, int verticalThreshold) {
    // 按照纵坐标排序中心点
    std::sort(points.begin(), points.end(), [](const cv::Point& a, const cv::Point& b) {
        return a.y < b.y;
    });

    std::vector<cv::Point> filteredPoints;

    // 筛选相近的点
    for (size_t i = 0; i < points.size() - 1; ++i) {
        if (std::abs(points[i].y - points[i + 1].y) < verticalThreshold) {
            // 纵坐标相差不多的点被认为是相近的点
            filteredPoints.push_back(points[i]);
        }
    }

    return filteredPoints;
}

//计算每个轮廓的最低点并返回点集中的最高点
cv::Point findHighestPointInContours(const std::vector<std::vector<cv::Point>>& contours, const cv::Mat& mask) {
    // 存储每个轮廓的最低点
    std::vector<cv::Point> lowestPoints;

    // 遍历每个轮廓
    for (const auto& contour : contours) {
        // 初始化最低点为轮廓中的第一个点
        cv::Point lowestPoint = contour[0];

        // 遍历轮廓中的每个点
        for (const auto& point : contour) {
            // 找到轮廓中的最低点
            if (point.y > lowestPoint.y) {
                lowestPoint = point;
            }
        }

        // 将最低点添加到存储最低点的容器中
        lowestPoints.push_back(lowestPoint);
    }

    // 找到存储最低点的容器中最高的点
    cv::Point highestPoint = lowestPoints[0];
    for (const auto& point : lowestPoints) {
        if (point.y < highestPoint.y) {
            highestPoint = point;
        }
    }

    return highestPoint;
}


// 寻找中间点的函数
std::vector<cv::Point> findMiddlePoints(const cv::Mat& lane_mask, int width, int height, int middle, int start_, int end_, int step) {
    std::vector<cv::Point> points;

    for (int i = start_; i < end_; i += step) {
        // 二分查找方法
        int left_point = -1;
        int right_point = -1;

        int left = middle;
        int right = middle - 1;
        while (left >= 0 && right_point == -1) {
            if (lane_mask.at<uchar>(i, left)) {
                left_point = left;
                break;
            }
            if (lane_mask.at<uchar>(i, right)) {
                right_point = right;
                break;
            }
            left--;
            right--;
        }

        left = middle + 1;
        right = middle;
        while (right < width && left_point == -1) {
            if (lane_mask.at<uchar>(i, right)) {
                right_point = right;
                break;
            }
            if (lane_mask.at<uchar>(i, left)) {
                left_point = left;
                break;
            }
            left++;
            right++;
        }

        if (left_point != -1 && right_point != -1) {
            cv::Point point;
            point.x = (left_point + right_point) / 2;
            point.y = i;
            points.push_back(point);
        }

    }

    return points;
}

//计算横坐标平均值
int calculateAverageX(const std::vector<cv::Point>& points) {
    if (points.empty()) {
        return 0; // 返回默认值或者您认为合适的值
    }

    int sumX = 0;

    // 遍历每个点，累积横坐标值
    for (const auto& point : points) {
        sumX += point.x;
    }

    // 计算平均值
    int averageX = static_cast<double>(sumX) / points.size();

    return averageX;
}

// 结构体用于存储中心点坐标和面积
struct RectangleInfo {
    cv::Point center;
    int area;
};

// 计算矩形框中心点和面积
std::vector<RectangleInfo> getRectangleInfo(const std::vector<YoloP::Box>& boxes) {
    std::vector<RectangleInfo> rectangleInfoVector;

    for (const auto& ibox : boxes) {
        // 计算中心点坐标
        int centerX = (ibox.left + ibox.right) / 2;
        int centerY = (ibox.top + ibox.bottom) / 2;

        // 计算矩形框面积
        int area = (ibox.right - ibox.left) * (ibox.bottom - ibox.top);

        // 创建 RectangleInfo 结构体并添加到向量中
        RectangleInfo rectangleInfo;
        rectangleInfo.center = cv::Point(centerX, centerY);
        rectangleInfo.area = area;
        rectangleInfoVector.push_back(rectangleInfo);
    }

    return rectangleInfoVector;
}

/*
* 打开串口
*/
int open_port(int com_port)
{
    int fd;
    /* 使用普通串口 */
    // TODO::在此处添加串口列表
    char* dev[] = { "/dev/ttyTHS0", "/dev/ttyUSB0" };
 
    //O_NDELAY 同 O_NONBLOCK。
    fd = open(dev[com_port], O_RDWR | O_NOCTTY);
    if (fd < 0)
    {
        perror("open serial port");
        return(-1);
    }
 
    //恢复串口为阻塞状态 
    //非阻塞：fcntl(fd,F_SETFL,FNDELAY)  
    //阻塞：fcntl(fd,F_SETFL,0) 
    if (fcntl(fd, F_SETFL, 0) < 0)
    {
        perror("fcntl F_SETFL\n");
    }
    /*测试是否为终端设备*/
    if (isatty(STDIN_FILENO) == 0)
    {
        perror("standard input is not a terminal device");
    }
 
    return fd;
}
 
/*
* 串口设置
*/
int set_uart_config(int fd, int baud_rate, int data_bits, char parity, int stop_bits)
{
    struct termios opt;
    int speed;
    if (tcgetattr(fd, &opt) != 0)
    {
        perror("tcgetattr");
        return -1;
    }
 
    /*设置波特率*/
    switch (baud_rate)
    {
    case 2400:  speed = B2400;  break;
    case 4800:  speed = B4800;  break;
    case 9600:  speed = B9600;  break;
    case 19200: speed = B19200; break;
    case 38400: speed = B38400; break;
    default:    speed = B115200; break;
    }
    cfsetispeed(&opt, speed);
    cfsetospeed(&opt, speed);
    tcsetattr(fd, TCSANOW, &opt);
 
    opt.c_cflag &= ~CSIZE;
 
    /*设置数据位*/
    switch (data_bits)
    {
    case 7: {opt.c_cflag |= CS7; }break;//7个数据位  
    default: {opt.c_cflag |= CS8; }break;//8个数据位 
    }
 
    /*设置奇偶校验位*/
    switch (parity) //N
    {
    case 'n':case 'N':
    {
        opt.c_cflag &= ~PARENB;//校验位使能     
        opt.c_iflag &= ~INPCK; //奇偶校验使能  
    }break;
    case 'o':case 'O':
    {
        opt.c_cflag |= (PARODD | PARENB);//PARODD使用奇校验而不使用偶校验 
        opt.c_iflag |= INPCK;
    }break;
    case 'e':case 'E':
    {
        opt.c_cflag |= PARENB;
        opt.c_cflag &= ~PARODD;
        opt.c_iflag |= INPCK;
    }break;
    case 's':case 'S': /*as no parity*/
    {
        opt.c_cflag &= ~PARENB;
        opt.c_cflag &= ~CSTOPB;
    }break;
    default:
    {
        opt.c_cflag &= ~PARENB;//校验位使能     
        opt.c_iflag &= ~INPCK; //奇偶校验使能          	
    }break;
    }
 
    /*设置停止位*/
    switch (stop_bits)
    {
    case 1: {opt.c_cflag &= ~CSTOPB; } break;
    case 2: {opt.c_cflag |= CSTOPB; }   break;
    default: {opt.c_cflag &= ~CSTOPB; } break;
    }
 
    /*处理未接收字符*/
    tcflush(fd, TCIFLUSH);
 
    /*设置等待时间和最小接收字符*/
    opt.c_cc[VTIME] = 1000;
    opt.c_cc[VMIN] = 0;
 
    /*关闭串口回显*/
    opt.c_lflag &= ~(ICANON | ECHO | ECHOE | ECHOK | ECHONL | NOFLSH);
 
    /*禁止将输入中的回车翻译为新行 (除非设置了 IGNCR)*/
    opt.c_iflag &= ~ICRNL;
    /*禁止将所有接收的字符裁减为7比特*/
    opt.c_iflag &= ~ISTRIP;
 
    /*激活新配置*/
    if ((tcsetattr(fd, TCSANOW, &opt)) != 0)
    {
        perror("tcsetattr");
        return -1;
    }
 
    return 0;
}


int main() {
    
    // 创建yolopv2检测器(速度慢，但检测效果好)
    auto detector = YoloP::create_detector("/home/rory/Projects/Linfer/workspace/yolopv2-480x640.trt", YoloP::Type::V2, 0, 0.4, 0.5);
    // 创建yolop检测器(速度快，但检测效果较差)
    //auto detector = YoloP::create_detector("/home/rory/Projects/Linfer/workspace/yolop-640.trt", YoloP::Type::V1, 0, 0.4, 0.5);
    // 预热处理20张图片以免启动时较慢影响判断
    auto image = cv::imread("/home/rory/Projects/Linfer/workspace/imgs/1.jpg");
    YoloP::PTMM res;
    for(int i = 0; i < 20; ++i)
        res = detector->detect(image);


    // begin::第一步，串口初始化
    int UART_fd = open_port(0);
	if (set_uart_config(UART_fd,115200, 8, 'E', 1) < 0)
	{
		perror("set_com_config");
		exit(1);
	}
    // end::串口初始化
    
    
    //USB摄像头
    //cv::VideoCapture cap(1);
    
    //CSI摄像头
    int capture_width = 1280 ;
    int capture_height = 720 ;
    int display_width = 1280 ;
    int display_height = 720 ;
    int framerate = 30 ;
    int flip_method = 0 ;

    std::string pipeline = gstreamer_pipeline(capture_width,
	capture_height,
	display_width,
	display_height,
	framerate,
	flip_method);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n";
 
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    //视频检测
    //cv::VideoCapture cap("/home/rory/Projects/Linfer/workspace/videos/video3.mp4");
    if (cap.isOpened()) {
        cv::namedWindow("CSI Camera", cv::WINDOW_AUTOSIZE);
        while (1) {
            
            // 开始检测
            auto start = std::chrono::steady_clock::now();
            cv::Mat img;
            cap >> img;

            //缩放为宽：640，高：480
            int width = 640;
            int height = 480;
            cv::resize(img,img, cv::Size(width, height));

            auto result = detector->detect(img); // 使用处理后的图像进行检测

            // 0对应车辆框，1对应驾驶区域，2对应车道线
            YoloP::BoxArray& boxes = get<0>(result);
            cv::Mat& drive_mask = get<1>(result);
            cv::Mat& lane_mask = get<2>(result);
    
    //判断是否行驶
    bool run = true;
    int area_threshold=width/5*height/4;

    std::vector<RectangleInfo> rectangleInfoVector;
    if (!boxes.empty()){
        // 获取矩形框信息（中心点坐标和面积）
        std::vector<RectangleInfo> rectangleInfoVector = getRectangleInfo(get<0>(result));
    }

    // 计算轮廓
    std::vector<std::vector<cv::Point>> drive_contours;
    cv::findContours(drive_mask, drive_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // 计算面积
    int drive_area = 0;
    for (const auto& drive_contour : drive_contours) {
        drive_area += cv::contourArea(drive_contour);
    }
    if (boxes.empty() && drive_area < area_threshold){
        run = false;
    }
    std::cout << run << std::endl;


    //转化为灰度图，将通道转化为1，不然后面图像处理有问题
    //cv::cvtColor(lane_mask, lane_mask, cv::COLOR_BGR2GRAY);

    //二值化
    int threshold_value = 128; // 设定阈值
    int max_value = 255; // 最大值（白色）
    cv::threshold(lane_mask, lane_mask, threshold_value, max_value, cv::THRESH_BINARY);

     //腐蚀
    cv::Mat kernel = cv::Mat::ones(4,5, CV_8U);
    cv::erode(lane_mask, lane_mask, kernel);

    //得到轮廓
    std::vector<std::vector<cv::Point>> laneContours;
    cv::findContours(lane_mask, laneContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    //计算得到轮廓中心点
    std::vector<cv::Point2f> laneCenters = calculateLaneCenters(laneContours);

    // 调用判断函数
    int imageCenterX = width/2;
    LanePosition position = checkLanePositions(laneCenters,imageCenterX);
    char str[10];
	int len = 9;

    // 处理返回结果
    switch (position) {
        case LanePosition::ALL_LEFT:
            std::cout << "所有车道线中心点都在图像左侧,应右转" << std::endl;
            // 遍历中心点向量
            // 遍历矩形框信息向量
            for (const auto& rectangleInfo : rectangleInfoVector) {
                // 检查中心点的 x 坐标是否位于右侧并且面积大于阈值
                if (rectangleInfo.center.x > imageCenterX && rectangleInfo.area > area_threshold) {
                    run = false;
                    break;  // 一旦找到符合条件的矩形框，就无需继续检查
                }
            }
            if(run){
            strcpy(str, "@:640,1\r\n");
            }
            else{
            strcpy(str, "@:640,0\r\n");
            }
            //发送数据
            write(UART_fd, str, len);
            cv::waitKey(1);
            break;


        case LanePosition::ALL_RIGHT:
            std::cout << "所有车道线中心点都在图像右侧,应左转" << std::endl;
            // 遍历中心点向量
            for (const auto& rectangleInfo : rectangleInfoVector) {
                // 检查中心点的x坐标是否位于右侧
                if (rectangleInfo.center.x < imageCenterX && rectangleInfo.area > area_threshold) {
                    run = false;
                    break;  // 一旦找到左侧的点，就无需继续检查
                }
            }
            if(run){
            strcpy(str, "@:000,1\r\n");
            }
            else{
            strcpy(str, "@:000,0\r\n");
            }
            //发送数据
            write(UART_fd, str, len);
            cv::waitKey(1);
            break;
        case LanePosition::MIXED:
            //std::cout << "车道线两侧都有中心点，进行计算缺口中心" << std::endl;

            //找到最顶部的点
            std::vector<cv::Point> topmostPoints = findTopmostPoints(laneContours);
            // 调用函数进行过滤
            int verticalThreshold = 10; // 调整阈值
            std::vector<cv::Point> filteredPoints = filterPointsByVertical(topmostPoints, verticalThreshold);

            // 计算横坐标平均值
            int sumX = 0;
            int sumY = 0;
            int averageX;
            int averageY;
            for (const auto& point : filteredPoints) {
                sumX += point.x;
                sumY += point.y;
            }
            // 确保有点，以避免除以零
            if (!filteredPoints.empty()) {
            averageX = sumX / filteredPoints.size();
            //std::cout << "横坐标平均值：" << averageX << std::endl;
            averageY = sumY / filteredPoints.size();
            //std::cout << "纵坐标平均值：" << averageY << std::endl;
            }else{
                averageX = width/2;
                averageY = height/2;
            }
            // 轮廓的最低点集中的最高点
            //cv::Point highestPoint = findHighestPointInContours(laneContours, lane_mask);
            //找到的两个点的中间点
            int middle = averageX;
            int start_ = averageY;
            int end_ = height;
            int step = 4;

            //开始遍历
            std::vector<cv::Point> points;
            points = findMiddlePoints(lane_mask, width, height, middle, start_, end_, step);
            for (const auto& point : filteredPoints) {
                sumX += point.x;
                sumY += point.y;
            }
            // 调用函数计算横坐标平均值
            int finallX = calculateAverageX(points);
            int finallY = averageY;
            if(finallX==0)
            {
                continue;
            }
            else{
            std::cout << "缺口坐标:(" << finallX << "," << finallY << ")" << std::endl;
            // 在图像上绘制点
            cv::circle(img, cv::Point(static_cast<int>(finallX), static_cast<int>(finallY)), 5, cv::Scalar(0, 0, 255), -1); // 红色点
            
            int gap_width=width/10;
            // 遍历中心点向量
            for (const auto& rectangleInfo : rectangleInfoVector) {
                // 检查中心点的x坐标是否位于右侧
                if ((finallX - gap_width < rectangleInfo.center.x) && (rectangleInfo.center.x < finallX + gap_width) && (rectangleInfo.area > area_threshold)) {
                    run = false;
                    break;  // 一旦找到左侧的点，就无需继续检查
                }
            }
            
            // 使用字符串流
            std::ostringstream oss;
            oss << "@:";

            if (finallX < 10) {
            // 如果 finallX 是个位数，在前面补充两个零
            oss << "00";
            } else if (finallX < 100) {
            // 如果 finallX 是十位数，在前面补充一个零
            oss << "0";
            }
            
            oss << finallX;
            if(run){
                oss << ",1\r\n";
            }
            else{
                oss << ",0\r\n";
            }
            // 获取生成的字符串
            std::string str_ = oss.str();
            strcpy(str, str_.c_str());
            write(UART_fd, str, len);
            cv::waitKey(1);
            }

            break;
    }



    //标注出车辆
    for(auto& ibox : boxes)
        cv::rectangle(img, cv::Point(ibox.left, ibox.top),
                      cv::Point(ibox.right, ibox.bottom),
                      {0, 0, 255}, 2);
    // 将可驾驶区域像素设置为绿色
    //img.setTo(cv::Scalar(0, 255, 0), drive_mask);

    //将白线区域像素设置成蓝色
    //img.setTo(cv::Scalar(255, 0, 0), lane_mask);


    // 获取帧率
    std::chrono::duration<double> during = std::chrono::steady_clock::now() - start;
    double use_time = 1000.0 * during.count();
    printf("FPS: %.2f\n", 1000 / use_time);


    cv::imshow("CSI Camera", img);
    //cv::imshow("lane_mask",lane_mask);
    int keyCode = cv::waitKey(1) & 0xFF;
    if (keyCode == 27)
    {  
        // ESC键退出
        break;
    }
        }

    cap.release();
    cv::destroyAllWindows();
    } else {
        std::cout << "打开摄像头失败" << std::endl;
    }

    return 0;
}


/*
// 使用实例
int main()
{
    // begin::第一步，串口初始化
    int UART_fd = open_port(0);
	if (set_uart_config(UART_fd,115200, 8, 'E', 1) < 0)
	{
		perror("set_com_config");
		exit(1);
	}
    // end::串口初始化
    
    // begin::第二步，读下位机上发的一行数据
    char str[128]="123abc";
	char buff[1];
	int len = 6;
	// while (1)
	// {
	// 	if (read(UART_fd, buff, 1)) {
	// 		if (buff[0] == '\n') {
	// 			break;
	// 		}else {
	// 			str[len++] = buff[0];
	// 		}
	// 	}
	// }
    while(1)
    {printf("content:%s\n",str);
    // end::读下位机上发的一行数据
 
    // begin::第三步，向下位机发送数据
    write(UART_fd, str, len);
    // end::向下位机发送数据
    cv::waitKey(5);
    }
    return 0;
}
*/