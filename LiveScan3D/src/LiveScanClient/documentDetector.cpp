/***************************************************************************\

Module Name:  DocumentDetector.cpp
Project:      LiveScan3D
Authors:      Roxanne Archambault
Copyright (c) Canadian Space Agency.

<Description>
This module uses a YOLO machine learning model to detect documents from a
provided color frame and ranks its detections based on their size and blur.

\***************************************************************************/

#include "documentDetector.h"

#include <algorithm>
#include <numeric>
#ifdef _WIN32
#include <Windows.h>
#endif


DocumentDetector::DocumentDetector()
{
    // Attempt to load the model early to detect any failure
    LoadModelIfNeeded();
    StartDetectionThread();
}

DocumentDetector::~DocumentDetector()
{
    // Stop the detection thread
    StopDetectionThread();
}

static std::basic_string<ORTCHAR_T> ToOrtString(const std::string& s) {
#ifdef _WIN32
    if (s.empty()) return std::basic_string<ORTCHAR_T>();
    int sizeNeeded = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), NULL, 0);
    std::wstring w(sizeNeeded, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], sizeNeeded);
    return w;
#else
    return s;
#endif
}

// Helper functions for Non-Maximum Suppression (NMS)
static float IoU(const cv::Rect& a, const cv::Rect& b) {
    int x1 = (std::max)(a.x, b.x);
    int y1 = (std::max)(a.y, b.y);
    int x2 = (std::min)(a.x + a.width, b.x + b.width);
    int y2 = (std::min)(a.y + a.height, b.y + b.height);
    int interArea = (std::max)(0, x2 - x1) * (std::max)(0, y2 - y1);
    int unionArea = a.area() + b.area() - interArea;

    return unionArea > 0 ? static_cast<float>(interArea) / unionArea : 0.0f;
}

// Helper function to perform Non-Maximum Suppression (NMS) on detected bounding boxes 
static void NMS(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, float iouThreshold, std::vector<int>& keep) {
    keep.clear();
    std::vector<int> idxs(boxes.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](int a, int b) { return scores[a] > scores[b]; });

    while (!idxs.empty()) {
        int best = idxs.front();
        keep.push_back(best);
        idxs.erase(idxs.begin());

        idxs.erase(std::remove_if(idxs.begin(), idxs.end(), [&](int i) { return IoU(boxes[best], boxes[i]) > iouThreshold; }), idxs.end());
    }
}

void DocumentDetector::SetDetectionCallback(DetectionCallback callback) {
    resultCallback = std::move(callback);
}

/// <summary>
/// Sets the logging function to be used to append messages to the logging file.
/// </summary>
/// <param name="loggerFunc">Function to be used for logging. Should be passed by orbbecCaptureManager.cpp.</param>
void DocumentDetector::SetLogger(std::function<void(const std::string&)> loggerFunc) {
    logFn = loggerFunc;
}

// Sets the ONNX model path and resets variables. Call before the first detection (or right after construction).
void DocumentDetector::ResetModel(const std::string& onnxPath) {
    modelPath = onnxPath;
    modelLoaded = false;
    ortSession.reset();
    inputNames.clear();
    outputNames.clear();
    inputNameStrs.clear();
    outputNameStrs.clear();
}

/// <summary>
/// Loads the ONNX model and prepares its inference session if it hasn't
/// already been loaded. The ONNX Runtime environment, session, and
/// input/output names are cached so subsequent calls can reuse them.
/// </summary>
/// <returns>True if the model is ready, false if loading fails</returns>
bool DocumentDetector::LoadModelIfNeeded() {
    // Already loaded and the session is valid — nothing to do
    if (modelLoaded && ortSession) return true;

    try {
        // Create the ONNX Runtime environment once
        // This is shared by the session and manages runtime resources
        if (!ortEnv) {
            ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DocumentDetector");
        }

        // Configure how ONNX Runtime will execute the model
        ortSessionOptions = Ort::SessionOptions();

        // Let ONNX Runtime apply all available graph optimization
        ortSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // CPU-only inference: use one thread for each type of work
        // so inference doesn't consume too many resources
        ortSessionOptions.SetIntraOpNumThreads(1);
        ortSessionOptions.SetInterOpNumThreads(1);

        // Convert the model path to the string type expected by
        // ONNX Runtime on the current platform
        auto ortPath = ToOrtString(modelPath);

        // Load the ONNX model and create the inference session
        ortSession = std::make_unique<Ort::Session>(*ortEnv, ortPath.c_str(), ortSessionOptions);

        Ort::AllocatorWithDefaultOptions allocator;

        // Cache input names
        size_t numInputs = ortSession->GetInputCount();
        inputNameStrs.clear();
        inputNames.clear();
        inputNameStrs.reserve(numInputs);
        inputNames.reserve(numInputs);

        for (size_t i = 0; i < numInputs; ++i) {
            // ONNX Runtime gives us an allocated name
            auto nameAllocated = ortSession->GetInputNameAllocated(i, allocator);

            // Copy it into std::string so the name remains alive
            inputNameStrs.emplace_back(nameAllocated.get());
        }

        // Build the const char* array expected by Run()
        // The strings themselves are owned by inputNameStrs
        for (auto& s : inputNameStrs) inputNames.push_back(s.c_str());


        // Cache output names
        size_t numOutputs = ortSession->GetOutputCount();
        outputNameStrs.clear();
        outputNames.clear();
        outputNameStrs.reserve(numOutputs);
        outputNames.reserve(numOutputs);

        for (size_t i = 0; i < numOutputs; ++i) {
            auto nameAllocated = ortSession->GetOutputNameAllocated(i, allocator);
            // Again, copy the name so we own the string's lifetime
            outputNameStrs.emplace_back(nameAllocated.get());
        }

        // Build the const char* array used when calling Run()
        for (auto& s : outputNameStrs) outputNames.push_back(s.c_str());

        modelLoaded = true;

        if (logFn) logFn(std::string("[DocumentDetector] Loaded ONNX model: ") + modelPath);
        return true;
    }
    catch (const Ort::Exception& e) {
        modelLoaded = false;

        if (logFn) logFn(std::string("[DocumentDetector] Failed to load ONNX model: ") + e.what());
        return false;
    }
}


/// <summary>
/// Submits a new frame for document detection.
/// </summary>
/// <param name="color">Color frame on which to perform the document detection</param>
/// <param name="depth">Depth frame on which to perform the document detection, aligned with the color frame</param>
void DocumentDetector::SubmitFrame(std::shared_ptr<ob::ColorFrame> color, cv::Mat depth)
{
    // Store the newly submitted frame in local variables for thread processing
    std::lock_guard<std::mutex> lock(frameMutex);
    pendingColorFrame = color;
    pendingDepthFrame = depth;
    newFrameAvailable = true;
    frameCond.notify_one();
}

/// <summary>
/// Starts the detection thread and detects document from provided frames
/// </summary>
void DocumentDetector::StartDetectionThread()
{
    stopThread = false;

    detectThread = std::thread([this]() {

        while (!stopThread)
        {
            std::shared_ptr<ob::ColorFrame> localColor;

            // Wait for new frame
            {
                std::unique_lock<std::mutex> lock(frameMutex);
                frameCond.wait(lock, [this]() { return newFrameAvailable || stopThread; });

                if (stopThread) break;

                // Store latest frame in local variables
                localColor = pendingColorFrame;

                newFrameAvailable = false;
            }

            // Try to detect a document from the frame
            cv::Mat data;
            float score = 0.0f;
            short width = 0, height = 0;
            bool found = Detect(localColor, data, width, height, score);

            // Call the detection callback if a document has been detected
            if (found && resultCallback) {
                DetectionResult result;
                result.data = std::move(data);
                result.width = width;
                result.height = height;
                result.score = score;

                resultCallback(result);
            }
        }
    });
}

void DocumentDetector::StopDetectionThread()
{
    {
        std::lock_guard<std::mutex> lock(frameMutex);
        stopThread = true;
        frameCond.notify_all();  // Wake the thread if waiting
    }

    if (detectThread.joinable())
        detectThread.join();
}


/// <summary>
/// Uses an ONNX YOLO model to detect any documents in the provided frame
/// </summary>
/// <param name="colorFrame">Color frame from the camera from which to detect documents</param>
/// <param name="documentData">Output pixels composing the detected document</param>
/// <param name="documentPictureWidth">Output width of the detected document, in pixels</param>
/// <param name="documentPictureHeight">Output height of the detected document, in pixels</param>
/// <param name="documentScore">Score of the detected document to compare it with other detections</param>
/// <returns>True if a document was detected, false otherwise</returns>
bool DocumentDetector::Detect(
    const std::shared_ptr<ob::ColorFrame>& colorFrame,
    cv::Mat& documentData,
    short& documentPictureWidth,
    short& documentPictureHeight,
    float& bestScore
)
{
    bestScore = 0.0f;

    if (!colorFrame || colorFrame->data() == nullptr) {
        return false;
    }

    if (!LoadModelIfNeeded()) {
        return false;
    }

    // Convert Orbbec color frame to OpenCV Mat (RGB)
    cv::Mat originalImage(colorFrame->height(), colorFrame->width(), CV_8UC3, (void*)colorFrame->data());

    LetterboxInfo lb;
    cv::Mat inputImage = PrepareInputImage(originalImage, lb);

    // Convert to float tensor NCHW, normalized to [0,1]
    std::vector<float> inputTensor;
    Ort::Value inputOrt{ nullptr };

    if (!CreateInputTensor(inputImage, inputTensor, inputOrt)) {
        return false;
    }

    // Inference
    std::vector<Ort::Value> outputs;
    if (!RunInference(inputOrt, outputs)) {
        return false;
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    if (!DecodeDetections(outputs, lb, originalImage, boxes, scores)) {
        return false;
    }

    // NMS
    // Removes overlapping detections so that multiple boxes
    // around the same document don't compete with each other
    std::vector<int> keep;
    NMS(boxes, scores, nmsThreshold, keep);

    if (keep.empty()) return false;

    // Choose best detection with a strong preference for bigger boxes (documents)
    const double imgArea = static_cast<double>(originalImage.cols) * static_cast<double>(originalImage.rows);
    bool found = false;

    for (int idx : keep) {
        const cv::Rect& box = boxes[idx];
        const float conf = scores[idx];

        const double areaRatio = static_cast<double>(box.area()) / imgArea;

        // Strongly prefer larger detections so we don't end up with tiny crops (e.g., 785px wide)
        const double score = 0.4 * conf + 0.6 * areaRatio;

        if (score > bestScore) {
            cv::Mat documentData;
            short documentPictureWidth, documentPictureHeight;

            if (!ExtractDocumentCrop(originalImage, box, documentData, documentPictureWidth, documentPictureHeight)) {
                continue;
            }

            bestScore = static_cast<float>(score);
            found = true;

            if (logFn) {
                logFn("[DocumentDetector] Selected crop: " +
                    std::to_string(documentPictureWidth) + "x" +
                    std::to_string(documentPictureHeight) +
                    " conf=" + std::to_string(conf) +
                    " areaRatio=" + std::to_string(areaRatio));
            }
        }
    }

    return found;
}

cv::Mat DocumentDetector::PrepareInputImage(cv::Mat& originalImage, LetterboxInfo& letterboxInfo)
{
    cv::cvtColor(originalImage, originalImage, cv::COLOR_BGR2RGB);

    // Resize the image while preserving its aspect ratio.
    // The remaining space is filled with gray padding so the
    // result is exactly 640x640.
    auto letterbox = [&](const cv::Mat& src, cv::Mat& dst, LetterboxInfo& info) {
        const int w = src.cols;
        const int h = src.rows;

        info.scale = (std::min)(static_cast<float>(kInputSize) / static_cast<float>(w),
            static_cast<float>(kInputSize) / static_cast<float>(h));
        info.newW = static_cast<int>(std::round(w * info.scale));
        info.newH = static_cast<int>(std::round(h * info.scale));

        cv::Mat resized;
        cv::resize(src, resized, cv::Size(info.newW, info.newH), 0, 0, cv::INTER_LINEAR);

        // Center the resized image inside the 640x640 canvas
        info.padW = (kInputSize - info.newW) / 2;
        info.padH = (kInputSize - info.newH) / 2;

        dst = cv::Mat(cv::Size(kInputSize, kInputSize), CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(dst(cv::Rect(info.padW, info.padH, info.newW, info.newH)));
        };

    cv::Mat inputImage;
    letterbox(originalImage, inputImage, letterboxInfo);

    return inputImage;
}

bool DocumentDetector::CreateInputTensor(
    const cv::Mat& inputImage,
    std::vector<float>& inputTensor,
    Ort::Value& inputOrt)
{
    inputTensor.resize(1 * 3 * kInputSize * kInputSize);

    for (int y = 0; y < kInputSize; ++y) {
        const cv::Vec3b* row = inputImage.ptr<cv::Vec3b>(y);

        for (int x = 0; x < kInputSize; ++x) {
            // inputImage is RGB
            const float r = row[x][0] / 255.0f;
            const float g = row[x][1] / 255.0f;
            const float b = row[x][2] / 255.0f;

            const int idx = y * kInputSize + x;
            inputTensor[0 * kInputSize * kInputSize + idx] = r;
            inputTensor[1 * kInputSize * kInputSize + idx] = g;
            inputTensor[2 * kInputSize * kInputSize + idx] = b;
        }
    }

    // Describe the tensor as:
    // [batch=1, channels=3, height=640, width=640]
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 4> inputShape = { 1, 3, kInputSize, kInputSize };

    inputOrt =  Ort::Value::CreateTensor<float>(
        memInfo,
        inputTensor.data(),
        inputTensor.size(),
        inputShape.data(),
        inputShape.size()
    );

    return true;
}

bool DocumentDetector::RunInference(const Ort::Value& input, std::vector<Ort::Value>& outputs)
{
    try {
        outputs = ortSession->Run(
            Ort::RunOptions{ nullptr },
            inputNames.data(),
            &input,
            1,
            outputNames.data(),
            outputNames.size()
        );
    }
    catch (const Ort::Exception& e) {
        if (logFn) logFn(std::string("[DocumentDetector] ONNX inference failed: ") + e.what());
        return false;
    }

    if (outputs.empty()) return false;

    return true;
}

bool DocumentDetector::DecodeDetections(
    const std::vector<Ort::Value>& outputs,
    const LetterboxInfo& lb,
    const cv::Mat& originalImage,
    std::vector<cv::Rect>& boxes,
    std::vector<float>& scores)
{
    // YOLOv8-seg ONNX typically returns 2 outputs:
    //  - output0: detection data, shape [1, C, N] or [1, N, C]
    //  - output1: segmentation prototypes (we ignore for bbox-only crop)
    const Ort::Value& det = outputs[0];
    auto detInfo = det.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> detShape = detInfo.GetShape();

    if (detShape.size() != 3) {
        if (logFn) logFn("[DocumentDetector] Unexpected detection output shape.");
        return false;
    }

    int64_t dim1 = detShape[1];
    int64_t dim2 = detShape[2];

    // Determine layout ([1, C, N] or [1, N, C])
    int64_t C = 0, N = 0;
    bool layoutCHW = true; // [1, C, N]

    if (dim1 < dim2) {
        C = dim1; N = dim2; layoutCHW = true;
    }
    else {
        C = dim2; N = dim1; layoutCHW = false; // [1, N, C]
    }

    const float* detData = det.GetTensorData<float>();

    if (!detData) return false;

    // If a prototype output exists, use it to infer mask dimension,
    // otherwise fall back to YOLOv8 default (32).
    int64_t maskDim = 32;

    if (outputs.size() >= 2) {
        auto protoInfo = outputs[1].GetTensorTypeAndShapeInfo();
        auto protoShape = protoInfo.GetShape(); // [1, maskDim, mh, mw]

        if (protoShape.size() >= 2 && protoShape[1] > 0) maskDim = protoShape[1];
    }

    int64_t clsCount = C - 4 - maskDim;

    if (clsCount <= 0) {
        // Some exports may omit mask coefficients in output0
        clsCount = C - 4;
        maskDim = 0;
    }

    // Helper to read one value regardless of whether the tensor
    // is [1,C,N] or [1,N,C]
    auto at = [&](int64_t i, int64_t c) -> float {
        if (layoutCHW) {
            // [1, C, N] contiguous as C-major
            return detData[c * N + i];
        }
        else {
            // [1, N, C]
            return detData[i * C + c];
        }
        };

    // Convert raw model detections into OpenCV bounding boxes
    boxes.reserve(static_cast<size_t>(N));
    scores.reserve(static_cast<size_t>(N));

    for (int64_t i = 0; i < N; ++i) {
        // YOLO bounding box format:
        // center X/Y + width/height
        float cx = at(i, 0);
        float cy = at(i, 1);
        float w = at(i, 2);
        float h = at(i, 3);

        // Find the highest class confidence for this detection
        float bestCls = 0.0f;

        for (int64_t c = 0; c < clsCount; ++c) {
            float s = at(i, 4 + c);

            if (s > bestCls) bestCls = s;
        }

        float conf = bestCls;

        if (conf < confThreshold) continue;

        // Convert center-based coordinates to corner coordinates
        float x1 = cx - w * 0.5f;
        float y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f;
        float y2 = cy + h * 0.5f;

        // Map from letterboxed 640x640 back to original image coordinates
        x1 = (x1 - static_cast<float>(lb.padW)) / lb.scale;
        y1 = (y1 - static_cast<float>(lb.padH)) / lb.scale;
        x2 = (x2 - static_cast<float>(lb.padW)) / lb.scale;
        y2 = (y2 - static_cast<float>(lb.padH)) / lb.scale;

        int left = (std::max)(0, static_cast<int>(std::floor(x1)));
        int top = (std::max)(0, static_cast<int>(std::floor(y1)));
        int right = (std::min)(originalImage.cols - 1, static_cast<int>(std::ceil(x2)));
        int bottom = (std::min)(originalImage.rows - 1, static_cast<int>(std::ceil(y2)));

        int width = right - left;
        int height = bottom - top;

        if (width <= 2 || height <= 2) continue;

        boxes.emplace_back(left, top, width, height);
        scores.emplace_back(conf);
    }

    if (boxes.empty()) return false;

    return true;
}

bool DocumentDetector::ExtractDocumentCrop(
    const cv::Mat& originalImage,
    const cv::Rect& box,
    cv::Mat& documentData,
    short& documentPictureWidth,
    short& documentPictureHeight)
{
    // Add padding around the bbox to capture more of the page (10%)
    int padX = static_cast<int>(box.width * 0.10f);
    int padY = static_cast<int>(box.height * 0.10f);

    cv::Rect padded(
        box.x - padX,
        box.y - padY,
        box.width + 2 * padX,
        box.height + 2 * padY
    );

    // Clamp to image bounds
    cv::Rect safeBox = padded & cv::Rect(0, 0, originalImage.cols, originalImage.rows);

    if (safeBox.width <= 0 || safeBox.height <= 0) return false;

    // Extract the document (enlarge for downstream processing)
    documentData = originalImage(safeBox).clone();
    cv::resize(documentData, documentData, cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);

    // Apply a mild unsharp-mask effect to improve readability
    cv::Mat blurred;
    cv::GaussianBlur(documentData, blurred, cv::Size(0, 0), 1.0);
    cv::addWeighted(documentData, 1.5, blurred, -0.5, 0, documentData);

    // Save a debug image every ~30 detections
    static int dbg = 0;

    if (dbg++ % 5 == 0) {
        cv::Mat bgr;
        cv::cvtColor(documentData, bgr, cv::COLOR_RGB2BGR);
        cv::imwrite("doc_debug.png", bgr);   // lossless
    }

    documentPictureWidth = static_cast<short>(documentData.cols);
    documentPictureHeight = static_cast<short>(documentData.rows);

    return true;
}