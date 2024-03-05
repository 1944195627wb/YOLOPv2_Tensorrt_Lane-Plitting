

#include "trt_common/ilogger.hpp"
#include "yolo/yolo.hpp"
#include <opencv2/opencv.hpp>
#include "bytetrack/BYTETracker.h"
#include <cstdio>

using namespace std;

template<typename Cond>
static vector<Object> det2tracks(const Yolo::BoxArray& array, const Cond& cond){

    vector<Object> outputs;
    for(int i = 0; i < array.size(); ++i){
        auto& abox = array[i];

        if(!cond(abox)) continue;

        Object obox;
        obox.prob = abox.confidence;
        obox.label = abox.label;
        obox.rect[0] = abox.left;
        obox.rect[1] = abox.top;
        obox.rect[2] = abox.right - abox.left;
        obox.rect[3] = abox.bottom - abox.top;
        outputs.emplace_back(obox);
    }
    return outputs;
}


void inference_bytetrack(const string& engine_file, int gpuid, Yolo::Type type, const string& video_file){

    auto engine = Yolo::create_infer(
            engine_file,                // engine file
            type,                       // yolo type, Yolo::Type::V5 / Yolo::Type::X
            gpuid,                   // gpu id
            0.25f,                      // confidence threshold
            0.45f,                      // nms threshold
            Yolo::NMSMethod::CUDA,   // NMS method, fast GPU / CPU
            1024,                       // max objects
            false                       // preprocess use multi stream
    );
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    cv::VideoCapture cap(video_file);
    auto fps = cap.get(cv::CAP_PROP_FPS);
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    BYTETracker tracker;
    cv::Mat image;
    cv::Mat prev_image;
    tracker.config().set_initiate_state({0.1,  0.1,  0.1,  0.1,
                                                0.2,  0.2,  1,    0.2}
                                        ).set_per_frame_motion({0.1,  0.1,  0.1,  0.1,
                                                                       0.2,  0.2,  1,    0.2}
                                        ).set_max_time_lost(150);

    cv::VideoWriter writer("videos/res_mot.mp4", cv::VideoWriter::fourcc('M', 'P', 'E', 'G'), fps, cv::Size(width, height));
    auto cond = [](const Yolo::Box& b){return b.label == 0;};

    shared_future<vector<Yolo::Box>> prev_fut;
    int t = 0;
    while(cap.read(image)){
        t++;
        /// 高性能的关键：
        /// 先缓存一帧图像，使得读图预处理和推理后处理的时序图有重叠
        if(prev_fut.valid()){
            const auto& boxes = prev_fut.get();
            auto tracks = tracker.update(det2tracks(boxes, cond));
            for(auto& track : tracks){

                vector<float> tlwh = track.tlwh;
                // 通过宽高比和面积过滤掉
                bool vertical = tlwh[2] / tlwh[3] > 1.6;
                if (tlwh[2] * tlwh[3] > 20 && !vertical)
                {
                    auto s = tracker.get_color(track.track_id);
                    putText(prev_image, cv::format("%d", track.track_id), cv::Point(tlwh[0], tlwh[1] - 10),
                            0, 2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
                    rectangle(prev_image, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]),
                              cv::Scalar(get<0>(s), get<1>(s), get<2>(s)), 3);
                }
            }
            writer.write(prev_image);
            printf("process.\n");
        }

        image.copyTo(prev_image);
        prev_fut = engine->commit(image);

    }

    writer.release();
    printf("Done.\n");
}

