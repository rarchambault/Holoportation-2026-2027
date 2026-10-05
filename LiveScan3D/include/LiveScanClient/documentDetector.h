/***************************************************************************\

Module Name:  DocumentDetector.h
Project:      LiveScan3D
Authors:      Roxanne Archambault
Copyright (c) Canadian Space Agency.

<Description>
This module uses a YOLO machine learning model to detect documents from a
provided color frame and ranks its detections based on their size and blur.

\***************************************************************************/

#pragma once
#include <opencv2/opencv.hpp>
#include "libobsensor/ObSensor.hpp"
#include <utils.h>
#include <vector>
#include <string>
#include <mutex>
#include <functional>
#include <memory>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <onnxruntime_cxx_api.h>

struct LetterboxInfo {
    float scale = 1.0f;
    int padW = 0;
    int padH = 0;
    int newW = 0;
    int newH = 0;
};

class DocumentDetector
{
public:
    // Type alias for the detection callback
    using DetectionCallback = std::function<void(const DetectionResult&)>;

    DocumentDetector();
    ~DocumentDetector();

    void SubmitFrame(std::shared_ptr<ob::ColorFrame> color, cv::Mat depth);

    bool Detect(
        const std::shared_ptr<ob::ColorFrame>& colorFrame,
        cv::Mat& documentData,
        short& documentPictureWidth,
        short& documentPictureHeight,
        float& documentScore
    );

    void SetDetectionCallback(DetectionCallback callback);
    void SetLogger(std::function<void(const std::string&)> loggerFunc);
    void ResetModel(const std::string& onnxPath);

private:
    std::mutex frameMutex;
    std::condition_variable frameCond;

    std::shared_ptr<ob::ColorFrame> pendingColorFrame = nullptr;
    cv::Mat pendingDepthFrame;

    int numBackgroundSamples = 0;
    int numRequiredBackgroundSamples = 5;
    std::vector<cv::Mat> backgroundDepthSamples;
    cv::Mat averageBackgroundDepth;

    bool newFrameAvailable = false;
    bool stopThread = false;
    std::thread detectThread;

    int stableCount_ = 0;
    cv::Point2f lastCenter_{ -1.f, -1.f };
    cv::Rect lastBox_{ 0,0,0,0 };

    DetectionCallback resultCallback;

    // Inference parameters
    const int kInputSize = 640;
    const float confThreshold = 0.25f;
    const float nmsThreshold = 0.45f;

    // YOLOv8-seg ONNX Runtime (CPU)
    std::string modelPath = "document_yolov8seg.onnx";
    std::unique_ptr<Ort::Env> ortEnv;
    std::unique_ptr<Ort::Session> ortSession;
    Ort::SessionOptions ortSessionOptions;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    std::vector<std::string> inputNameStrs;
    std::vector<std::string> outputNameStrs;
    bool modelLoaded = false;

    bool LoadModelIfNeeded();
    void StartDetectionThread();
    void StopDetectionThread();
    cv::Mat PrepareInputImage(cv::Mat& originalImage, LetterboxInfo& letterboxInfo);
    bool CreateInputTensor(
        const cv::Mat& inputImage,
        std::vector<float>& inputTensor,
        Ort::Value& inputOrt);
    bool RunInference(const Ort::Value& input, std::vector<Ort::Value>& outputs);
    bool DecodeDetections(
        const std::vector<Ort::Value>& outputs,
        const LetterboxInfo& lb,
        const cv::Mat& originalImage,
        std::vector<cv::Rect>& boxes,
        std::vector<float>& scores);
    bool ExtractDocumentCrop(
        const cv::Mat& originalImage,
        const cv::Rect& box,
        cv::Mat& documentData,
        short& documentPictureWidth,
        short& documentPictureHeight);
    std::function<void(const std::string&)> logFn;
};