#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"
#include <JetsonGPIO.h>

#include <iostream>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;

static Logger gLogger;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

bool parse_args(int argc, char **argv, std::string &wts, std::string &engine, bool &is_p6, float &gd, float &gw, std::string &img_dir)
{
  if (argc < 4)
    return false;
  if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7))
  {
    wts = std::string(argv[2]);
    engine = std::string(argv[3]);
    auto net = std::string(argv[4]);
    if (net[0] == 'n')
    {
      gd = 0.33;
      gw = 0.25;
    }
    else if (net[0] == 's')
    {
      gd = 0.33;
      gw = 0.50;
    }
    else if (net[0] == 'm')
    {
      gd = 0.67;
      gw = 0.75;
    }
    else if (net[0] == 'l')
    {
      gd = 1.0;
      gw = 1.0;
    }
    else if (net[0] == 'x')
    {
      gd = 1.33;
      gw = 1.25;
    }
    else if (net[0] == 'c' && argc == 7)
    {
      gd = atof(argv[5]);
      gw = atof(argv[6]);
    }
    else
    {
      return false;
    }
    if (net.size() == 2 && net[1] == '6')
    {
      is_p6 = true;
    }
  }
  else if (std::string(argv[1]) == "-d" && argc == 4)
  {
    engine = std::string(argv[2]);
    img_dir = std::string(argv[3]);
  }
  else
  {
    return false;
  }
  return true;
}

void prepare_buffers(ICudaEngine *engine, float **gpu_input_buffer, float **gpu_output_buffer, float **cpu_output_buffer)
{
  assert(engine->getNbBindings() == 2);
  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int inputIndex = engine->getBindingIndex(kInputTensorName);
  const int outputIndex = engine->getBindingIndex(kOutputTensorName);
  assert(inputIndex == 0);
  assert(outputIndex == 1);
  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc((void **)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)gpu_output_buffer, kBatchSize * kOutputSize * sizeof(float)));

  *cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

void infer(IExecutionContext &context, cudaStream_t &stream, void **gpu_buffers, float *output, int batchsize)
{
  context.enqueue(batchsize, gpu_buffers, stream, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}

void serialize_engine(unsigned int max_batchsize, bool &is_p6, float &gd, float &gw, std::string &wts_name, std::string &engine_name)
{
  // Create builder
  IBuilder *builder = createInferBuilder(gLogger);
  IBuilderConfig *config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an engine
  ICudaEngine *engine = nullptr;
  if (is_p6)
  {
    engine = build_det_p6_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
  }
  else
  {
    engine = build_det_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
  }
  assert(engine != nullptr);

  // Serialize the engine
  IHostMemory *serialized_engine = engine->serialize();
  assert(serialized_engine != nullptr);

  // Save engine to file
  std::ofstream p(engine_name, std::ios::binary);
  if (!p)
  {
    std::cerr << "Could not open plan output file" << std::endl;
    assert(false);
  }
  p.write(reinterpret_cast<const char *>(serialized_engine->data()), serialized_engine->size());

  // Close everything down
  engine->destroy();
  config->destroy();
  serialized_engine->destroy();
  builder->destroy();
}

void deserialize_engine(std::string &engine_name, IRuntime **runtime, ICudaEngine **engine, IExecutionContext **context)
{
  std::ifstream file(engine_name, std::ios::binary);
  if (!file.good())
  {
    std::cerr << "read " << engine_name << " error!" << std::endl;
    assert(false);
  }
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  char *serialized_engine = new char[size];
  assert(serialized_engine);
  file.read(serialized_engine, size);
  file.close();

  *runtime = createInferRuntime(gLogger);
  assert(*runtime);
  *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
  assert(*engine);
  *context = (*engine)->createExecutionContext();
  assert(*context);
  delete[] serialized_engine;
}

struct
{
  cv::Rect box;
  bool drawing = false;
} globalData;

// 鼠标回调函数
void mouseCallback(int event, int x, int y, int flags, void *userdata)
{
  if (event == cv::EVENT_LBUTTONDOWN)
  {
    globalData.drawing = true;
    globalData.box = cv::Rect(x, y, 0, 0);
  }
  else if (event == cv::EVENT_MOUSEMOVE)
  {
    if (globalData.drawing)
    {
      globalData.box.width = x - globalData.box.x;
      globalData.box.height = y - globalData.box.y;
    }
  }
  else if (event == cv::EVENT_LBUTTONUP)
  {
    globalData.drawing = false;
  }
}

// 缩放和填充图像
std::tuple<cv::Mat, float, cv::Point> resizeAndPadImage(const cv::Mat &input, int target_width, int target_height)
{
  // 计算缩放因子和新尺寸
  double h1 = target_width * static_cast<double>(input.rows) / input.cols;
  double w2 = target_height * static_cast<double>(input.cols) / input.rows;
  double scale = std::min(target_width / static_cast<double>(input.cols), target_height / static_cast<double>(input.rows));

  int scaled_height, scaled_width;
  if (h1 <= target_height)
  {
    scaled_height = static_cast<int>(h1);
    scaled_width = target_width;
  }
  else
  {
    scaled_width = static_cast<int>(w2);
    scaled_height = target_height;
  }

  cv::Mat resized;
  cv::resize(input, resized, cv::Size(scaled_width, scaled_height), 0, 0, cv::INTER_LINEAR);

  // 创建填充图像
  int top = (target_height - scaled_height) / 2;
  int bottom = target_height - scaled_height - top;
  int left = (target_width - scaled_width) / 2;
  int right = target_width - scaled_width - left;

  cv::Mat padded;
  cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

  // 返回缩放因子和偏移量
  float newscale = std::min(target_width / static_cast<float>(scaled_width),
                            target_height / static_cast<float>(scaled_height));
  cv::Point offset(left, top);

  return std::make_tuple(padded, scale, offset);
}

int main(int argc, char **argv)
{

  // 设置 GPIO 编号
  int output_pin = 7; // 例如，GPIO7
  GPIO::setmode(GPIO::BOARD);
  GPIO::setup(output_pin, GPIO::OUT, GPIO::LOW);

  cudaSetDevice(kGpuId);

  std::string wts_name = "";
  std::string engine_name = "";
  bool is_p6 = false;
  float gd = 0.0f, gw = 0.0f;
  std::string img_dir;

  std::chrono::time_point<std::chrono::system_clock> last_detection_time;
  bool recently_detected = false;

  if (!parse_args(argc, argv, wts_name, engine_name, is_p6, gd, gw, img_dir))
  {
    std::cerr << "arguments not right!" << std::endl;
    std::cerr << "./yolov5_det -s [.wts] [.engine] [n/s/m/l/x/n6/s6/m6/l6/x6 or c/c6 gd gw]  // serialize model to plan file" << std::endl;
    std::cerr << "./yolov5_det -d [.engine] ../images  // deserialize plan file and run inference" << std::endl;
    return -1;
  }

  // Create a model using the API directly and serialize it to a file
  if (!wts_name.empty())
  {
    serialize_engine(kBatchSize, is_p6, gd, gw, wts_name, engine_name);
    return 0;
  }

  // Deserialize the engine from file
  IRuntime *runtime = nullptr;
  ICudaEngine *engine = nullptr;
  IExecutionContext *context = nullptr;
  deserialize_engine(engine_name, &runtime, &engine, &context);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Init CUDA preprocessing
  cuda_preprocess_init(kMaxInputImageSize);

  // Prepare cpu and gpu buffers
  float *gpu_buffers[2];
  float *cpu_output_buffer = nullptr;
  prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);

  // Read images from directory
  std::vector<std::string> file_names;
  if (read_files_in_dir(img_dir.c_str(), file_names) < 0)
  {
    std::cerr << "read_files_in_dir failed." << std::endl;
    return -1;
  }

  // batch predict
  for (size_t i = 0; i < file_names.size(); i += kBatchSize)
  {
    // Get a batch of images
    std::vector<cv::Mat> img_batch;
    std::vector<std::string> img_name_batch;
    for (size_t j = i; j < i + kBatchSize && j < file_names.size(); j++)
    {
      cv::Mat img = cv::imread(img_dir + "/" + file_names[j]);
      img_batch.push_back(img);
      img_name_batch.push_back(file_names[j]);
    }

    // Preprocess
    cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

    // Run inference
    auto start = std::chrono::system_clock::now();
    infer(*context, stream, (void **)gpu_buffers, cpu_output_buffer, kBatchSize);
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // NMS
    std::vector<std::vector<Detection>> res_batch;
    batch_nms(res_batch, cpu_output_buffer, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);

    // Draw bounding boxes
    draw_bbox(img_batch, res_batch);

    // Save images
    for (size_t j = 0; j < img_batch.size(); j++)
    {
      cv::imwrite("_" + img_name_batch[j], img_batch[j]);
    }
  }

  // 修改这里，使用V4L2作为后端
  cv::VideoCapture capture(0, cv::CAP_V4L2);
  if (!capture.isOpened())
  {
    std::cout << "Error opening video stream or file" << std::endl;
    return -1;
  }
  int key;
  int fcount = 0;
  bool objectDetected = false;
  // 主循环
  while (1)
  {
    cv::Mat frame;
    capture >> frame;
    if (frame.empty())
    {
      std::cout << "Fail to read image from camera!" << std::endl;
      break;
    }

    // 创建一个遮罩图像，初始化为黑色
    cv::Mat masked_img = cv::Mat::zeros(frame.size(), frame.type());

    // 如果定义了有效的矩形框，则只复制该矩形框内的图像区域
    if (!globalData.box.empty())
    {
      cv::Rect roi = globalData.box & cv::Rect(0, 0, frame.cols, frame.rows);
      frame(roi).copyTo(masked_img(roi));
    }
    else
    {
      frame.copyTo(masked_img);
    }

    // Preprocess
    cuda_preprocess(masked_img.ptr(), masked_img.cols, masked_img.rows, gpu_buffers[0], kInputW, kInputH, stream);

    // Run inference
    auto start = std::chrono::system_clock::now(); // 获取模型推理开始时间

    infer(*context, stream, (void **)gpu_buffers, cpu_output_buffer, kBatchSize);
    auto end = std::chrono::system_clock::now(); // 结束时间
    int fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::vector<Detection> batch_res;
    auto &res = batch_res;
    nms(res, cpu_output_buffer, kConfThresh, kNmsThresh);

    // 检查当前帧是否检测到物体
    objectDetected = !res.empty(); // 在每一帧中检测物体
    if (objectDetected)
    {
      last_detection_time = std::chrono::system_clock::now(); // 更新最后检测到物体的时间
      recently_detected = true;
      GPIO::output(output_pin, GPIO::HIGH); // 检测到物体，设置高电平
    }
    else if (recently_detected && (std::chrono::system_clock::now() - last_detection_time) > std::chrono::seconds(1))
    {
      recently_detected = false;
      GPIO::output(output_pin, GPIO::LOW); // 一秒后未检测到物体，设置低电平
    }

    // 在原始帧上绘制检测到的物体和框
    if (!globalData.box.empty())
    {
      cv::rectangle(frame, globalData.box, cv::Scalar(255, 0, 0), 2); // 绘制鼠标绘制的框
    }

    // 绘制检测结果
    for (size_t j = 0; j < res.size(); j++)
    {
      cv::Rect r = get_rect(masked_img, res[j].bbox); // 在遮罩图像上的坐标

      cv::rectangle(frame, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
      // 打印检测框的坐标
      std::cout << "Detected object " << j << ": ["
                << "x=" << r.x << ", y=" << r.y
                << ", width=" << r.width << ", height=" << r.height
                << "]" << std::endl;
    }

    std::string jetson_fps = "FPS: " + std::to_string(fps);
    cv::putText(frame, jetson_fps, cv::Point(11, 80), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

    // 创建一个可缩放的窗口
    cv::namedWindow("yolov5", cv::WINDOW_NORMAL);
    cv::setMouseCallback("yolov5", mouseCallback); // 设置鼠标回调
    cv::imshow("yolov5", frame);
    key = cv::waitKey(1);
    if (key == 'q')
    {
      break;
    }
    fcount = 0;
  }

  capture.release();
  // Release stream and buffers
  cudaStreamDestroy(stream);
  CUDA_CHECK(cudaFree(gpu_buffers[0]));
  CUDA_CHECK(cudaFree(gpu_buffers[1]));
  delete[] cpu_output_buffer;
  cuda_preprocess_destroy();
  // Destroy the engine
  context->destroy();
  engine->destroy();
  runtime->destroy();

  // Print histogram of the output distribution
  // std::cout << "\nOutput:\n\n";
  // for (unsigned int i = 0; i < kOutputSize; i++) {
  //   std::cout << prob[i] << ", ";
  //   if (i % 10 == 0) std::cout << std::endl;
  // }
  // std::cout << std::endl;
  GPIO::cleanup();

  return 0;
}
