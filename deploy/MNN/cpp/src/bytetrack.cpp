// #if defined(USE_NCNN_SIMPLEOCV)
// #include "simpleocv.h"
// #include <opencv2/opencv.hpp>
// #else
#include "MNN/Tensor.hpp"
#include "MNN/ImageProcess.hpp"
#include "MNN/Interpreter.hpp"
#include "MNN/MNNForwardType.h"
#include "MNN/MNNDefine.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <float.h>
#include <stdio.h>
#include <memory>
#include <dlfcn.h>
#include <vector>
#include <sys/time.h>
#include <chrono>
#include "BYTETracker.h"

#define YOLOX_NMS_THRESH 0.7  // nms threshold
#define YOLOX_CONF_THRESH 0.1 // threshold of bounding box prob
// #define INPUT_W 1088          // target image size w after resize
// #define INPUT_H 608           // target image size h after resize
// #define INPUT_W 640          // target image size w after resize
// #define INPUT_H 640           // target image size h after resize
#define INPUT_W 416          // target image size w after resize
#define INPUT_H 416           // target image size h after resize

using namespace std;
using namespace cv;
using namespace MNN;
using namespace MNN::CV;
// using namespace MNN::Express;

Mat static_resize(Mat &img)
{
    float r = min(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    Mat re(unpad_h, unpad_w, CV_8UC3);
    resize(img, re, re.size());
    Mat out(INPUT_H, INPUT_W, CV_8UC3, Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

// YOLOX use the same focus in yolov5
// class YoloV5Focus : public ncnn::Layer
// {
// public:
//     YoloV5Focus()
//     {
//         one_blob_only = true;
//     }

//     virtual int forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob, const ncnn::Option &opt) const
//     {
//         int w = bottom_blob.w;
//         int h = bottom_blob.h;
//         int channels = bottom_blob.c;

//         int outw = w / 2;
//         int outh = h / 2;
//         int outc = channels * 4;

//         top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
//         if (top_blob.empty())
//             return -100;

// #pragma omp parallel for num_threads(opt.num_threads)
//         for (int p = 0; p < outc; p++)
//         {
//             const float *ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
//             float *outptr = top_blob.channel(p);

//             for (int i = 0; i < outh; i++)
//             {
//                 for (int j = 0; j < outw; j++)
//                 {
//                     *outptr = *ptr;

//                     outptr += 1;
//                     ptr += 2;
//                 }

//                 ptr += w;
//             }
//         }

//         return 0;
//     }
// };

// DEFINE_LAYER_CREATOR(YoloV5Focus)

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static inline float intersection_area(const Object &a, const Object &b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j)
                qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right)
                qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object> &objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object &a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object &b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int> &strides, std::vector<GridAndStride> &grid_strides)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

static void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, const MNN::Tensor &feat_blob, float prob_threshold, std::vector<Object> &objects)
{
    int batches, channels, height, width, pred_items;
    batches = feat_blob.shape()[0];
    channels = feat_blob.shape()[1];
    height = feat_blob.shape()[2];
    width = feat_blob.shape()[3];
    pred_items = feat_blob.shape()[4];
    int fffff = feat_blob.shape()[5];
    // cout<<"grid size: "<<grid_strides.size()<<"feature.h "<<feat_blob.h<<endl;
    cout << "batchs: " << batches << " channels: " << channels << "  height: " << height << "  width:" << width << " pred_items: " << pred_items << "   " << fffff << feat_blob.shape()[6] << "  " << feat_blob.shape()[7] << " " << feat_blob.shape()[8] << endl;
    int num_classes = 2;
    auto data_ptr = feat_blob.host<float>();
    // for (int anchor_idx = 0; anchor_idx < grid_strides.size(); anchor_idx++)
    // {
    //     const int grid0 = grid_strides[anchor_idx].grid0;
    //     const int grid1 = grid_strides[anchor_idx].grid1;
    //     const int stride = grid_strides[anchor_idx].stride;

    //     float x_center = (data_ptr[0] + grid0) * stride;
    //     float y_center = (data_ptr[1] + grid1) * stride;
    //     float w = exp(data_ptr[2]) * stride;
    //     float h = exp(data_ptr[3]) * stride;
    //     float x0 = x_center - w * 0.5f;
    //     float y0 = y_center - h * 0.5f;

    //     float box_objectness = data_ptr[4];
    //     for (int class_idx = 0; class_idx < num_classes; class_idx++)
    //     {
    //         float box_cls_score =data_ptr[5 + class_idx];
    //         float box_prob = box_objectness * box_cls_score;
    //         if (box_prob > prob_threshold)
    //         {
    //             Object obj;
    //             obj.rect.x = x0;
    //             obj.rect.y = y0;
    //             obj.rect.width = w;
    //             obj.rect.height = h;
    //             obj.label = class_idx;
    //             obj.prob = box_prob;

    //             objects.push_back(obj);
    //         }
    //     }
    //     data_ptr += width;
    // }
    vector<vector<float>> raw_loc_res;
    vector<vector<float>> raw_prob_res;
    for (int i = 0; i < grid_strides.size(); i++)
    {
        vector<float> temp_raw_loc_res;
        vector<float> temp_raw_prob_res;
        for (int j = 0; j < 4; j++)
        {
            temp_raw_loc_res.push_back(*data_ptr);
            data_ptr++;
        }
        for (int j = 0; j < num_classes; j++)
        {
            temp_raw_prob_res.push_back(*data_ptr);
            data_ptr++;
        }
        raw_loc_res.push_back(temp_raw_loc_res);
        raw_prob_res.push_back(temp_raw_prob_res);
    }
    for (int i = 0; i < grid_strides.size(); i++)
    {
        const int grid0 = grid_strides[i].grid0;
        const int grid1 = grid_strides[i].grid1;
        const int stride = grid_strides[i].stride;
        raw_loc_res[i][0] = (raw_loc_res[i][0] + grid0) * stride;
        raw_loc_res[i][1] = (raw_loc_res[i][1] + grid1) * stride;
        raw_loc_res[i][2] = exp(raw_loc_res[i][2]) * stride;
        raw_loc_res[i][3] = exp(raw_loc_res[i][3]) * stride;
    }

    vector<float> score;
    vector<vector<float>> bbox;
    for (int i = 0; i < num_classes - 1; i++)
    {
        for (int j = 0; j < grid_strides.size(); j++)
        {
            score.push_back(raw_prob_res[j][i + 1]);
            bbox.push_back(raw_loc_res[j]);
        }
    }
    for (int i = 0; i < grid_strides.size() * (num_classes - 1); i++)
    {
        if (score[i] > prob_threshold)
        {
            Object obj;
            obj.rect.x = bbox[i][0] - 0.5 * bbox[i][2];

            obj.rect.y = bbox[i][1] - 0.5 * bbox[i][3];
            obj.rect.width = bbox[i][2];
            obj.rect.height = bbox[i][3];
            obj.label = i / grid_strides.size();
            obj.prob = score[i];

            objects.push_back(obj);
        }
    }
}
// static int detect_yolox(MNN::Session *session, std::vector<Object> &objects, shared_ptr<MNN::Interpreter> net, float scale, shared_ptr<MNN::CV::ImageProcess> pretreat_data)
static int detect_yolox(MNN::Tensor &output_host, std::vector<Object> &objects, float scale, int dw_, int dh_, int img_width, int img_height)
{
    //cout << "batchs: " <<output_host.shape()[0] << " channels: " << output_host.shape()[1] << "  height: " << output_host.shape()[2]<< "  width:" <<output_host.shape()[3] << " pred_items: " << output_host.shape()[4] << "   " << output_host.shape()[5] << "  " << output_host.shape()[6] << " " << output_host.shape()[8] << endl;
    auto pred_dims = output_host.shape();
    const unsigned int num_anchors = pred_dims.at(1);
    const unsigned int num_classes = pred_dims.at(2) - 5;
    // auto ptr = output_host.host<float>();
    std::vector<Object> proposals;

    static const int stride_arr[] = {8, 16, 32}; // might have stride=64 in YOLOX
    std::vector<int> strides(stride_arr, stride_arr + sizeof(stride_arr) / sizeof(stride_arr[0]));
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(INPUT_W, INPUT_H, strides, grid_strides); // grid_strides=anchor
    // generate_yolox_proposals(grid_strides, output_host, YOLOX_CONF_THRESH, proposals);
    cout<<"num anchors: "<<num_anchors<<" grid_strides "<<grid_strides.size()<<endl;
    unsigned int num_count = 0;
    int nn = 0;
    for (int i = 0; i < num_anchors; i++)
    {
        const float *offset_obj_cls_ptr = output_host.host<float>() + (i * (num_classes + 5));
        float obj_conf = offset_obj_cls_ptr[4];
        cout<<"obj_conf: "<<obj_conf<<endl;
        cout<<offset_obj_cls_ptr[0]<<"  "<<offset_obj_cls_ptr[1]<<"  "<<offset_obj_cls_ptr[2]<<" "<<offset_obj_cls_ptr[3]<<"  "<<offset_obj_cls_ptr[4]<<"  "<<offset_obj_cls_ptr[5]<<endl;
        nn += 1;
        
        if (obj_conf < 0.25f)
            continue;
        // if (obj_conf < 0.00000025f)
        //     continue;
        cout<<"bigger than threshold"<<endl;
        float cls_conf = offset_obj_cls_ptr[5];
        unsigned int label = 0;
        for (int j = 0; j < num_classes; j++)
        {
            float tmp_conf = offset_obj_cls_ptr[j + 5];
            //cout<<"obj_conf: "<<obj_conf<<" cls_score: "<<tmp_conf<<" box_prob: "<<box_prob<<endl;
            if (tmp_conf > cls_conf)
            {
                cls_conf = tmp_conf;
                label = j;
            }
        }
        float conf = obj_conf * cls_conf; // cls_conf (0.,1.)
        // if (conf < YOLOX_CONF_THRESH)
        //     continue;
        if (conf < 0.25f)
            continue;
        const int grid0 = grid_strides.at(i).grid0;
        const int grid1 = grid_strides.at(i).grid1;
        const int stride = grid_strides.at(i).stride;
        // float dx = offset_obj_cls_ptr[0];
        // float dy = offset_obj_cls_ptr[1];
        // float dw = offset_obj_cls_ptr[2];
        // float dh = offset_obj_cls_ptr[3];
        // float cx = (dx + (float) grid0) * (float) stride;
        // float cy = (dy + (float) grid1) * (float) stride;
        // float w = std::exp(dw) * (float) stride;
        // float h = std::exp(dh) * (float) stride;
        // float x1 = ((cx - w / 2.f) - (float) dw_) /scale;
        // float y1 = ((cy - h / 2.f) - (float) dh_) / scale;
        // float x2 = ((cx + w / 2.f) - (float) dw_) / scale;
        // float y2 = ((cy + h / 2.f) - (float) dh_) / scale;
        float x_center = (offset_obj_cls_ptr[0]+grid0)*stride;
        float y_center = (offset_obj_cls_ptr[1]+grid1)*stride;
        float w = exp(offset_obj_cls_ptr[2])*stride;
        float h = exp(offset_obj_cls_ptr[3])*stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;
        Object obj;
        obj.rect.x =  x0;
        obj.rect.y =   y0;
        obj.rect.width =  w;
        obj.rect.height =  h;

        obj.prob = conf;
        obj.label = label;
        proposals.push_back(obj);
        //cout<<obj.rect.width<<"  "<<obj.rect.height<<"  "<<obj.rect.x<<"  "<<obj.rect.y<<endl;
        num_count += 1; // limit boxes for nms.
        if (num_count > 30000)
        break;

    }
    cout<<"proposal size "<<proposals.size()<<endl;
    // sort all proposals by score from highest to lowest
    //cout<<"proposals.size() "<<proposals.size()<<endl;
    qsort_descent_inplace(proposals);
    //cout<<"proposals size after qsort_descent"<<proposals.size()<<endl;
    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, YOLOX_NMS_THRESH);

    int count = picked.size();

    objects.resize(count);
    // objects.resize(proposals.size());
    for (int i = 0; i < count; i++)
    // for (int i = 0; i < proposals.size(); i++)
    {
        objects[i] = proposals[picked[i]];
        //objects[i] = proposals[i];
        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x) / scale;
        float y0 = (objects[i].rect.y) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

        // clip
        // x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        // y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        // x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        // y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
    //cout<<"objects.size: "<<objects.size()<<endl;
    return 0;
}



// int main(int argc, char **argv)
// {
//     const char *class_names[80] = {
//         "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
//         "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
//         "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
//         "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
//         "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
//         "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
//         "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
//         "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
//         "scissors", "teddy bear", "hair drier", "toothbrush"
//     };
//     if (argc != 2)
//     {
//         fprintf(stderr, "Usage: %s [videopath]\n", argv[0]);
//         return -1;
//     }

//     // ncnn::Net yolox;
//     string model_path = "/home/shenzhang/Desktop/ByteTrack/yolox_s.mnn";
//     //string model_path = "/home/shenzhang/Desktop/ByteTrack/bytetrack_tiny.mnn";
//     std::shared_ptr<MNN::Interpreter> net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
//     if (net == nullptr)
//     {
//         cout << "Create interprete failed" << endl;
//         return -1;
//     }
//     ScheduleConfig schedule_config;
//     schedule_config.type = MNN_FORWARD_CPU;
//     // schedule_config.numThread = 1;
//     auto session = net->createSession(schedule_config);
//     auto input = net->getSessionInput(session, "inputs");
//     cout << "session created" << endl;
//     const char *videopath = argv[1];

//     VideoCapture cap(videopath);
//     if (!cap.isOpened())
//     {
//         cout << "can not open video" << endl;
//         return 0;
//     }

//     int img_w = cap.get(CV_CAP_PROP_FRAME_WIDTH);
//     int img_h = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
//     int fps = cap.get(CV_CAP_PROP_FPS);
//     long nFrame = static_cast<long>(cap.get(CV_CAP_PROP_FRAME_COUNT));
//     cout << "Total frames: " << nFrame << endl;

//     VideoWriter writer("demo.mp4", CV_FOURCC('m', 'p', '4', 'v'), fps, Size(img_w, img_h));

//     Mat img;
//     BYTETracker tracker(fps, 30);
//     int num_frames = 0;
//     int total_ms = 1;

//     // get input data size

//     Matrix trans;
//     //trans.setScale(1.0f, 1.0f);
//     ImageProcess::Config config;
//     // config.filterType = CV::BILINERA;
//     const float mean_vals[3] = {255.f * 0.485f, 255.f * 0.456, 255.f * 0.406f};
//     const float norm_vals[3] = {1 / (255.f * 0.229f), 1 / (255.f * 0.224f), 1 / (255.f * 0.225f)};
//     ::memcpy(config.mean, mean_vals, sizeof(mean_vals));
//     ::memcpy(config.normal, norm_vals, sizeof(norm_vals));
//     config.sourceFormat = BGR;
//     config.destFormat = RGB;
//     std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config));
//     pretreat->setMatrix(trans);
//     cout << "finish init!" << endl;
//     //***********测试图片***************************
//     // cv::Mat img = cv::imread("/home/shenzhang/Desktop/ByteTrack/heyan.png");
//     // float scale = min(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));
//     // Matrix trans;
//     // //trans.setScale(1.0f, 1.0f);
//     // ImageProcess::Config config;
//     // // config.filterType = CV::BILINERA;
//     // const float mean_vals[3] = {255.f * 0.485f, 255.f * 0.456, 255.f * 0.406f};
//     // const float norm_vals[3] = {1 / (255.f * 0.229f), 1 / (255.f * 0.224f), 1 / (255.f * 0.225f)};
//     // ::memcpy(config.mean, mean_vals, sizeof(mean_vals));
//     // ::memcpy(config.normal, norm_vals, sizeof(norm_vals));
//     // config.sourceFormat = BGR;
//     // config.destFormat = RGB;
//     // std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config));
//     // pretreat->setMatrix(trans);
//     // cout << "finish init!" << endl;
//     // Mat pr_img = static_resize(img);
//     // // in_pad.substract_mean_normalize(mean_vals, norm_vals);
//     // pretreat->convert(pr_img.data, INPUT_H, INPUT_W, 0, input); // convert to tensor
//     // int dw = (INPUT_W - scale* img.cols)/2;
//     // int dh =  (INPUT_H - scale* img.rows)/2;
//     // net->runSession(session);
//     // MNN::Tensor *outputTensor = net->getSessionOutput(session, "outputs");

//     // // copy to host
//     // MNN::Tensor output_host(outputTensor, outputTensor->getDimensionType()); // NCHW
//     // outputTensor->copyToHostTensor(&output_host);

//     // std::vector<Object> objects;
//     // detect_yolox(output_host, objects, scale, dw, dh, 0, 0);
//     // cout<<objects.size()<<endl;
//     // for (int i = 0; i < objects.size(); i++)
//     // {
//     //     putText(img, format("%s", class_names[objects[i].label]), cv::Point(objects[i].rect.x-2, objects[i].rect.y- 5),
//     //                  1, 3, Scalar(25, 44, 212), 2, LINE_AA);
//     //     rectangle(img, cv::Rect(objects[i].rect.x, objects[i].rect.y, objects[i].rect.width, objects[i].rect.height), cv::Scalar(255, 255, 0), 2);

//     // }
//     // cv::imwrite("imge.jpg", img);


//     //************************************
//     for (;;)
//     {
//         if (!cap.read(img))
//             break;
//         num_frames++;
//         if (num_frames % 20 == 0)
//         {
//             cout << "Processing frame " << num_frames << " (" << num_frames * 1000000 / total_ms << " fps)" << endl;
//         }
//         if (img.empty())
//             break;
//         float scale = min(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));
//         Mat pr_img = static_resize(img);
//         // in_pad.substract_mean_normalize(mean_vals, norm_vals);
//         pretreat->convert(pr_img.data, INPUT_H, INPUT_W, 0, input); // convert to tensor
//         int dw = (INPUT_W - scale* img.cols)/2;
//         int dh =  (INPUT_H - scale* img.rows)/2;
//         net->runSession(session);
//         MNN::Tensor *outputTensor = net->getSessionOutput(session, "outputs");

//         // copy to host
//         MNN::Tensor output_host(outputTensor, outputTensor->getDimensionType()); // NCHW
//         outputTensor->copyToHostTensor(&output_host);

//         std::vector<Object> objects;
//         auto start = chrono::system_clock::now();


//         detect_yolox(output_host, objects, scale, dw, dh, img_w, img_h);

//         vector<STrack> output_stracks = tracker.update(objects);
//         cout<<"output_stracks.size "<<output_stracks.size()<<endl;
//         auto end = chrono::system_clock::now();
//         total_ms = total_ms + chrono::duration_cast<chrono::microseconds>(end - start).count();
//         for (int i = 0; i < output_stracks.size(); i++)
//         {
//             vector<float> tlwh = output_stracks[i].tlwh;
//             bool vertical = tlwh[2] / tlwh[3] > 1.6;
//             if (tlwh[2] * tlwh[3] > 20 && !vertical)
//             {
//                 Scalar s = tracker.get_color(output_stracks[i].track_id);
//                 putText(img, format("%d", output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1] - 5),
//                         0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
//                 rectangle(img, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
//             }
//         }

//         cout<<"objects.size: "<<objects.size()<<endl;
//         // for (int i = 0; i < objects.size(); i++)
//         // {
//         //     cout<<objects[i].rect.x<<" "<<objects[i].rect.y<<" "<<objects[i].rect.width<<" "<<objects[i].rect.height<<endl;
//         //     rectangle(img, cv::Rect(objects[i].rect.x, objects[i].rect.y, objects[i].rect.width, objects[i].rect.height), cv::Scalar(255, 255, 0), 2);
//         // }

//         putText(img, format("frame: %d fps: %d num: %d", num_frames, num_frames * 1000000 / total_ms, output_stracks.size()),
//                 cv::Point(0, 30), 0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);

//         writer.write(img);
//         char c = waitKey(1);
//         if (c > 0)
//         {
//             break;
//         }
//     }
//     cap.release();
//     cout << "FPS: " << num_frames * 1000000 / total_ms << endl;
//     // ******************************************************


//     return 0;
// }




int main(int argc, char **argv)
{
    const char *class_names[80] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }
    auto handle = dlopen("libMNN_CL.so", RTLD_NOW);
    //FUNC_PRINT_ALL(handle, p);

    // ncnn::Net yolox;
    string model_path = "/home/shenzhang/Desktop/ByteTrack/yolox_tiny.mnn";
    //string model_path = "/home/linaro/zhangshen/ByteTrack/bytetrack_tiny.mnn";
    std::shared_ptr<MNN::Interpreter> net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
    if (net == nullptr)
    {
        cout << "Create interprete failed" << endl;
        return -1;
    }
    ScheduleConfig schedule_config;
    schedule_config.type = MNN_FORWARD_OPENCL;
    // schedule_config.numThread = 1;
    auto session = net->createSession(schedule_config);
    auto input = net->getSessionInput(session, NULL);
    cout << "session created" << endl;
    const char *image_path = argv[1];
    vector<String> img_files;
    cv::glob(image_path, img_files);
    if (img_files.size() == 0)
    {
      std::cout << "not found image" << std::endl;
      return -1;
    }

    BYTETracker tracker(1, 1);
    // get input data size

    Matrix trans;
    //trans.setScale(1.0f, 1.0f);
    ImageProcess::Config config;
    // config.filterType = CV::BILINERA;
    // const float mean_vals[3] = {255.f * 0.485f, 255.f * 0.456, 255.f * 0.406f};
    // const float norm_vals[3] = {1 / (255.f * 0.229f), 1 / (255.f * 0.224f), 1 / (255.f * 0.225f)};
    // ::memcpy(config.mean, mean_vals, sizeof(mean_vals));
    // ::memcpy(config.normal, norm_vals, sizeof(norm_vals));
    config.sourceFormat = BGR;
    config.destFormat = RGB;
    std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config));
    pretreat->setMatrix(trans);
    cout << "finish init!" << endl;
    
    int total_ms = 1;
    //************************************
    struct timeval time_start, time_end;
    for (int img_idx = 0; img_idx < img_files.size(); img_idx++)
    {
        cv::Mat img = cv::imread(img_files[img_idx], 1);
        int img_w = img.cols;
        int img_h = img.rows;
        
        if (img.empty())
        {
            cout<<"img is empty!"<<endl;
            break;
        }
        float scale = min(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));
        cout<<INPUT_H<<"  "<<INPUT_W<<endl;
        Mat pr_img = static_resize(img);
        // cv::imshow("dddddd", pr_img);
        // cv::waitKey();
        cout<<pr_img.cols<<"  "<<pr_img.rows<<endl;
        // in_pad.substract_mean_normalize(mean_vals, norm_vals);
        pretreat->convert(pr_img.data, INPUT_H, INPUT_W, 0, input); // convert to tensor
        int dw = (INPUT_W - scale* img.cols)/2;
        int dh =  (INPUT_H - scale* img.rows)/2;
        gettimeofday(&time_start, NULL);
        net->runSession(session);
        MNN::Tensor *outputTensor = net->getSessionOutput(session, NULL);
       
        // copy to host
        MNN::Tensor output_host(outputTensor, outputTensor->getDimensionType()); // NCHW
        outputTensor->copyToHostTensor(&output_host);
        gettimeofday(&time_end, NULL);
        auto usedTime = (time_end.tv_sec-time_start.tv_sec)*1000 + (time_end.tv_usec-time_start.tv_usec) / 1000;
        cout<<"per image cost "<<to_string(usedTime)<<" ms"<<endl;
        std::vector<Object> objects;
        

        detect_yolox(output_host, objects, scale, dw, dh, img_w, img_h);
        cout<<"objects size: "<<objects.size()<<endl;
        vector<STrack> output_stracks = tracker.update(objects);
        cout<<"output_stracks.size "<<output_stracks.size()<<endl;
        
        for (int i = 0; i < output_stracks.size(); i++)
        {
            vector<float> tlwh = output_stracks[i].tlwh;
            bool vertical = tlwh[2] / tlwh[3] > 1.6;
            if (tlwh[2] * tlwh[3] > 20 && !vertical)
            {
                Scalar s = tracker.get_color(output_stracks[i].track_id);
                putText(img, format("%d", output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1] - 5),
                        0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
                rectangle(img, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
            }
        }
        cv::imwrite("/home/shenzhang/Desktop/ByteTrack/deploy/MNN/cpp/result/"+to_string(img_idx)+".jpg", img);
    }

    return 0;
}

