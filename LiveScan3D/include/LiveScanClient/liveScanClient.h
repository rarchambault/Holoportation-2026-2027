/***************************************************************************\

Module Name:  LiveScanClient.h
Project:      LiveScan3D
Authors:      Roxanne Archambault
Copyright (c) Canadian Space Agency.

<Description>
This module handles all logic related to retrieving data from one camera and
setting its parameters. It also sends data back to the C# LiveScanServer.

This code was adapted from the following research:
Kowalski, M.; Naruniec, J.; Daniluk, M.: "LiveScan3D: A Fast and Inexpensive
3D Data Acquisition System for Multiple Kinect v2 Sensors". in 3D Vision (3DV),
2015 International Conference on, Lyon, France, 2015

\***************************************************************************/

#pragma once

#define WIN32_LEAN_AND_MEAN
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#define _WINSOCKAPI_

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#include "liveScanClientWrapper.h"
#include "resource.h"
#include "calibration.h"
#include "orbbecCaptureManager.h"
#include "frameIOHandler.h"
#include "transferObjectUtils.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <voxelGridFilter.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/poisson.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <json.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <unordered_map>

class LiveScanClient
{
public:
    LiveScanClientWrapper* wrapper = nullptr;

    LiveScanClient(int index);
    ~LiveScanClient();

    void Run();
    void StartFrameRecording();
    void Calibrate();
    void SetSettings(const CameraSettings& settings);
    void RequestRecordedFrame();
    void RequestLatestFrame();
    void RequestLatestMesh();
    void ReceiveCalibration(const AffineTransform& transform);
    void ClearRecordedFrames();
    void EnableSync(int syncState, int syncOffset);
    void DisableSync();
    void StartMaster();
    void RequestExit();

    std::function<void(const std::string&)> GetLogger();

private:
    const float Range = 0.3f;
    const float HalfRange = Range / 2.0f;
    const float MinPrecision = Range / 255; // Min precision (max resolution) with the set range and the number of values in a byte (255)
    const int GridResolution = Range / MinPrecision;

    const float XRangeCenter = 0.0f;
    const float YRangeCenter = 0.0f;
    const float ZRangeCenter = HalfRange;

    const float DocumentDiffThreshold = 0.50;
    const int DocumentSendTimeout = 30000; // In milliseconds

    int clientIndex = -1;
    bool isClientThreadRunning;

    bool isCalibrateRequested;
    bool isRecordFrameRequested;
    bool isConfirmRecordedRequested;
    bool isConfirmSyncStateRequested;
    bool isConfirmRestartAsMasterRequested;
    bool isConfirmCalibratedRequested;
    bool isSendDocumentRequested;
    
    bool isFilterEnabled;
    int numFilterNeighbors;
    float filterThreshold;

    bool isAutoExposureEnabled;
    int numExposureSteps;

    bool isRestartingCamera;

    volatile bool isExitRequested = false;

    SyncState currentSyncState;

    ICaptureManager* captureManager;
    Calibration calibration;
    VoxelGridFilter voxelGridFilter;
    FrameIOHandler framesFileWriterReader;

    std::vector<float> bounds;

    std::vector<Point3s> lastFrameVertices;
    std::vector<RGB> lastFrameColors;
    std::vector<float> lastFrameMeshVertices;
    std::vector<int> lastFrameMeshIndices;
    std::mutex dataMutex;
    int frameCounter = 0;


    cv::Mat lastDocumentData;
    float lastDocumentScore;
    short lastDocumentWidth;
    short lastDocumentHeight;
    std::chrono::milliseconds lastDocumentSendTime;

    std::thread processingThread;
    std::mutex frameMutex;
    std::condition_variable frameCV;
    bool hasNewFrameToProcess = false;

    std::vector<Point3f> rawBufferVertices;
    std::vector<RGB> rawBufferColors;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::PointCloud<pcl::Normal>::Ptr normals;
    pcl::PointCloud<pcl::PointNormal>::Ptr cloudWithNormals;
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr normalTree;
    pcl::PolygonMesh mesh;

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;

    std::vector<Point3f> localVertices;
    std::vector<RGB> localColors;

    Point3f* cameraSpaceCoordinates;

    std::ofstream logFile;

    void UpdateFrame();
    void ApplyVoxelDownsampling(std::vector<Point3f> &goodVertices, std::vector<RGB>&goodColorPoints);
    bool ConstructMesh(std::vector<Point3f> &goodVertices, std::vector<int> &tempMeshIndices);
    void ProcessDocument();
    float ComputeImageDifference(cv::Mat& newDocumentData);
    void SendSerialNumber();
    void ConfirmRecorded();
    void ConfirmCalibrated();
    void SendLatestFrame();
    void SendLatestMesh();
    void SendRecordedFrame(vector<Point3s>& vertices, vector<RGB>& RGB, bool noMoreFrames);
    void ConfirmSyncState();
    void ConfirmMasterRestart();
    void SendDocument();
    void SendClientConfirmations();
    void SetupLogging(int clientIndex);
    void ProcessingLoop();
    void Log(const std::string& message);
    void OutputFrameToJson(std::vector<Point3s> vertices, std::vector<RGB> colors, std::vector<int> meshIndices);
};
