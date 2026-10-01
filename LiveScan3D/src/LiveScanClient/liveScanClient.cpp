/***************************************************************************\

Module Name:  LiveScanClient.cpp
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

#include "stdafx.h"
#include "resource.h"
#include "liveScanClient.h"
#include "filter.h"
#include <chrono>
#include <strsafe.h>
#include <fstream>
#include <shellapi.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

LiveScanClient::LiveScanClient(int index) :
	clientIndex(index),
	cameraSpaceCoordinates(NULL),
	isCalibrateRequested(false),
	isFilterEnabled(false),
	isRecordFrameRequested(false),
	isConfirmRecordedRequested(false),
	isConfirmCalibratedRequested(false),
	isConfirmRestartAsMasterRequested(false),
	isClientThreadRunning(true),
	numFilterNeighbors(10),
	filterThreshold(0.01f),
	isRestartingCamera(false),
	isAutoExposureEnabled(true),
	numExposureSteps(-5),
	voxelGridFilter(MinPrecision, XRangeCenter, YRangeCenter, ZRangeCenter, HalfRange)
{
	SetupLogging(clientIndex);

	captureManager = new OrbbecCaptureManager(clientIndex);
	captureManager->SetLogger(GetLogger());
	calibration.SetLogger(GetLogger());

	bounds.push_back(-0.5);
	bounds.push_back(-0.5);
	bounds.push_back(-0.5);
	bounds.push_back(0.5);
	bounds.push_back(0.5);
	bounds.push_back(0.5);

	cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
	normals = std::make_shared<pcl::PointCloud<pcl::Normal>>();
	cloudWithNormals = std::make_shared<pcl::PointCloud<pcl::PointNormal>>();
	tree = std::make_shared<pcl::search::KdTree<pcl::PointNormal>>();
	normalTree = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();

	ne.setSearchMethod(normalTree);

	gp3.setSearchRadius(0.015f);
	gp3.setMu(2.5f);
	gp3.setMaximumNearestNeighbors(100);
	gp3.setMaximumSurfaceAngle(M_PI / 4);
	gp3.setMinimumAngle(M_PI / 36);
	gp3.setMaximumAngle(5 * M_PI / 6);
	gp3.setNormalConsistency(false);
	gp3.setSearchMethod(tree);
}

LiveScanClient::~LiveScanClient()
{
	if (captureManager)
	{
		delete captureManager;
		captureManager = NULL;
	}

	if (cameraSpaceCoordinates)
	{
		delete[] cameraSpaceCoordinates;
		cameraSpaceCoordinates = NULL;
	}
}

/// <summary>
/// Initializes the camera and starts the main loop to retrieve its data
/// </summary>
void LiveScanClient::Run()
{
	// First initialize the camera as standalone (sync disabled)
	bool res = captureManager->Initialize(Standalone, 0);

	if (res)
	{
		SendSerialNumber();

		// Try to load calibration data from a previous run
		calibration.LoadCalibration(captureManager->serialNumber);

		if (calibration.isCalibrated)
			isConfirmCalibratedRequested = true;

		cameraSpaceCoordinates = new Point3f[captureManager->colorFrameWidth * captureManager->colorFrameHeight];
		captureManager->SetExposureState(true, 0);
	}
	else
	{
		Log("[LiveScanClient] Failed to initialize capture device.");
	}

	// Start a thread to handle some client callbacks to the server in parallel to the main data loop
	std::thread t1(&LiveScanClient::SendClientConfirmations, this);

	processingThread = std::thread(&LiveScanClient::ProcessingLoop, this);

	// Start the main loop to retrieve data from the camera
	while (!isExitRequested)
	{
		UpdateFrame();
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}

	isClientThreadRunning = false;
	frameCV.notify_all();
	t1.join();
	if (processingThread.joinable()) processingThread.join();
}

void LiveScanClient::StartFrameRecording()
{
	isRecordFrameRequested = true;
}

void LiveScanClient::Calibrate()
{
	isCalibrateRequested = true;
}

void LiveScanClient::SetSettings(const CameraSettings& settings)
{
	bounds = { settings.MinBounds[0], settings.MinBounds[1], settings.MinBounds[2],
				  settings.MaxBounds[0], settings.MaxBounds[1], settings.MaxBounds[2] };

	isFilterEnabled = settings.Filter;
	numFilterNeighbors = settings.FilterNeighbors;
	filterThreshold = settings.FilterThreshold;

	// Copy marker poses to calibration data
	calibration.markerPoses.resize(settings.NumMarkers);

	for (int i = 0; i < settings.NumMarkers; i++) {
		calibration.markerPoses[i].MarkerId = settings.MarkerPoses[i].MarkerId;
		memcpy(calibration.markerPoses[i].R, settings.MarkerPoses[i].R, sizeof(float) * 9);
		memcpy(calibration.markerPoses[i].T, settings.MarkerPoses[i].T, sizeof(float) * 3);
	}

	isAutoExposureEnabled = settings.AutoExposureEnabled;
	numExposureSteps = settings.ExposureStep;

	captureManager->SetExposureState(isAutoExposureEnabled, numExposureSteps);
}

void LiveScanClient::RequestRecordedFrame()
{
	// Read the first recorded frame saved during recording
	vector<Point3s> points;
	vector<RGB> colors;
	bool res = framesFileWriterReader.ReadFrame(points, colors);

	SendRecordedFrame(points, colors, !res);
}

void LiveScanClient::RequestLatestFrame()
{
	SendLatestFrame();
}

void LiveScanClient::RequestLatestMesh()
{
	SendLatestMesh();
}

void LiveScanClient::ReceiveCalibration(const AffineTransform& transform)
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
			calibration.worldR[i][j] = transform.R[i][j];

		calibration.worldT[i] = transform.T[i];
	}
}

void LiveScanClient::ClearRecordedFrames()
{
	framesFileWriterReader.CloseFile();
}

void LiveScanClient::EnableSync(int syncState, int syncOffset)
{
	bool res = false;

	switch (syncState)
	{
	case 0:
		currentSyncState = Subordinate;
		isRestartingCamera = true;

		// Close camera
		res = captureManager->Close();
		if (!res) {
			Log("[LiveScanClient] Subordinate device failed to close! Restart Application!");
			return;
		}

		// Re-initialize as Subordinate with a unique syncOffset (sent by the server)
		res = captureManager->Initialize(Subordinate, syncOffset);
		if (!res) {
			Log("[LiveScanClient] Subordinate device failed to reinitialize! Restart Application!");
			return;
		}

		// Confirm reinitialization as Subordinate to the server
		isConfirmSyncStateRequested = true;
		isRestartingCamera = false;
		break;

	case 1:
		currentSyncState = Master;
		isRestartingCamera = true;

		// Close camera; need to wait until all Subordinates have reinitialized before restarting the Master
		res = captureManager->Close();
		if (!res) {
			Log("[LiveScanClient] Master device failed to close! Restart Application!");
			return;
		}

		// Confirm reinitialization as Master to the server
		isConfirmSyncStateRequested = true;
		break;

	case 2:
		currentSyncState = Standalone;
		isRestartingCamera = true;

		// Close camera
		res = captureManager->Close();
		if (!res) {
			Log("[LiveScanClient] Capture device failed to close! Restart Application!");
			return;
		}

		// Re-initialize as Standalone
		res = captureManager->Initialize(Standalone, 0);

		if (!res) {
			Log("[LiveScanClient] Capture device failed to reinitialize! Restart Application!");
			return;
		}

		// Confirm reinitialization as Standalone to the server
		isConfirmSyncStateRequested = true;
		isRestartingCamera = false;
		break;

	default:
		break;
	}
}

void LiveScanClient::DisableSync()
{
	// Set this device as Standalone
	currentSyncState = Standalone;
	isRestartingCamera = true;

	bool res;

	// Close the camera
	res = captureManager->Close();
	if (!res) {
		Log("[LiveScanClient] Capture device failed to close! Restart Application!");
		return;
	}

	// Re-initialize as Standalone
	res = captureManager->Initialize(Standalone, 0);

	if (!res) {
		Log("[LiveScanClient] Capture device failed to reinitialize! Restart Application!");
		return;
	}

	// Confirm reinitialization as Standalone to the server
	isConfirmSyncStateRequested = true;
	isRestartingCamera = false;
}

void LiveScanClient::StartMaster()
{
	// This is called by the server once all Subordinates have been re-initialized, meaning the Master can now start
	if (currentSyncState == Master)
	{
		bool res = captureManager->Initialize(Master, 0);
		if (!res) {
			Log("[LiveScanClient] Master device failed to reinitialize! Restart Application!");
			return;
		}

		isConfirmRestartAsMasterRequested = true;
		isRestartingCamera = false;
	}
}

void LiveScanClient::RequestExit()
{
	isExitRequested = true;
}

void LiveScanClient::SendClientConfirmations()
{
	while (isClientThreadRunning)
	{
		if (isConfirmRecordedRequested)
		{
			ConfirmRecorded();
		}

		if (isConfirmCalibratedRequested)
		{
			ConfirmCalibrated();
		}

		if (isConfirmSyncStateRequested)
		{
			ConfirmSyncState();
		}

		if (isConfirmRestartAsMasterRequested)
		{
			ConfirmMasterRestart();
		}

		if (isSendDocumentRequested)
		{
			SendDocument();
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}

/// <summary>
/// Retrieves point cloud data from the camera and stores it into local variables for sending to the server
/// </summary>
void LiveScanClient::UpdateFrame()
{
	// Check that the capture manager is initialized
	if (!captureManager->isInitialized)
	{
		return;
	}

	// Acquire a new point cloud frame from the camera
	bool newFrameAcquired = captureManager->AcquireFrame(isCalibrateRequested);

	if (!newFrameAcquired)
	{
		return;
	}

	// Apply some processing to the data that was just retrieved and store it in local variables

	if (!isCalibrateRequested)
	{
		{
			std::lock_guard<std::mutex> lock(frameMutex);
			// Copy raw data to the shared swap buffer
			rawBufferVertices = captureManager->lastFrameVertices;
			rawBufferColors = captureManager->lastFrameColors;
			hasNewFrameToProcess = true;
		}
		// Wake up the meshing thread
		frameCV.notify_one();
	}

	// Process the document data from the frame
	if (captureManager->hasNewDocument) 
	{
		ProcessDocument();
		captureManager->hasNewDocument = false;
	}
	

	if (isRecordFrameRequested)
	{
		// If we are recording frames, save the frame that was just processed
		uint64_t timeStamp = captureManager->GetTimeStamp();
		std::lock_guard<std::mutex> lock(dataMutex);
		framesFileWriterReader.WriteFrame(lastFrameVertices, lastFrameColors, timeStamp, captureManager->GetDeviceIndex());

		isConfirmRecordedRequested = true;
		isRecordFrameRequested = false;
	}

	if (isCalibrateRequested)
	{
		// Calibrate the camera by using the marker(s) and their positions as specified in the settings
		int totalPixels = captureManager->depthFrameWidth * captureManager->depthFrameHeight;
		Point3f* floatPoints = new Point3f[totalPixels];
		RGB* colors = new RGB[totalPixels];

		for (int i = 0; i < totalPixels; i++) {
			floatPoints[i].X = captureManager->lastFrameVertices[i].X;
			floatPoints[i].Y = captureManager->lastFrameVertices[i].Y;
			floatPoints[i].Z = captureManager->lastFrameVertices[i].Z;

			colors[i].Red = captureManager->lastFrameColors[i].Red;
			colors[i].Green = captureManager->lastFrameColors[i].Green;
			colors[i].Blue = captureManager->lastFrameColors[i].Blue;
		} 

		bool res = calibration.Calibrate(colors, floatPoints, captureManager->depthFrameWidth, captureManager->depthFrameHeight);

		delete[] floatPoints;
		delete[] colors;

		if (res)
		{
			// Save the new calibration to a file to reuse in a later run
			calibration.SaveCalibration(captureManager->serialNumber);
			isConfirmCalibratedRequested = true;
			isCalibrateRequested = false;
		}
	}
}
/// <summary>
/// Applies calibration and filtering to the latest frame and creates a mesh
/// </summary>
void LiveScanClient::ProcessingLoop()
{
	while (isClientThreadRunning)
	{
		// Copy latest frame into local variables
		{
			std::unique_lock<std::mutex> lock(frameMutex);
			frameCV.wait(lock, [this] { return hasNewFrameToProcess || !isClientThreadRunning; });

			if (!isClientThreadRunning) break;

			localVertices = rawBufferVertices;
			localColors = rawBufferColors;
			hasNewFrameToProcess = false;
		}

		std::vector<Point3f> goodVertices;
		std::vector<RGB> goodColorPoints;
		ApplyVoxelDownsampling(goodVertices, goodColorPoints);

		// If the more complex filtering step is enabled, apply it now
		if (isFilterEnabled) Filter(goodVertices, goodColorPoints, numFilterNeighbors, filterThreshold);

		// Convert the remaining vertices to shorts to save memory
		vector<Point3s> goodVerticesShort(goodVertices.size());
		for (size_t i = 0; i < goodVertices.size(); i++) goodVerticesShort[i] = goodVertices[i];

		std::vector<int> tempMeshIndices;
		if (!ConstructMesh(goodVertices, tempMeshIndices)) continue;

		// Store data in shared variables for other threads to use
		{
			std::lock_guard<std::mutex> lock(dataMutex);
			lastFrameVertices = goodVerticesShort;
			lastFrameColors = goodColorPoints;
			lastFrameMeshIndices = tempMeshIndices;
		}

		// Uncomment this to output frames to JSON format
		//if (frameCounter % 100 == 0)
		//{
		//	OutputFrameToJson(goodVerticesShort, goodColorPoints, tempMeshIndices);
		//}

		//frameCounter++;
	}
}

void LiveScanClient::ApplyVoxelDownsampling(std::vector<Point3f> &goodVertices, std::vector<RGB> &goodColorPoints)
{
	// To save some processing cost, we allocate a full frame size (numVertices) of a Point3f Vector beforehand
	// instead of using push_back for each vertex. Even though we have to copy the vertices into a clean array
	// later and it uses a little bit more RAM, this gives us a nice speed increase for this function, around 25-50%.
	unsigned int numVertices = localVertices.size();
	vector<Point3f> allVertices(numVertices);

	// Create a placeholder for a point to remove
	Point3f invalidPoint = Point3f(0, 0, 0, true);

	const float densityVoxelSize = 0.006f;
	const int minPointsPerVoxel = 12;

	// Count points per voxel
	std::unordered_map<uint64_t, int> voxelCounts;
	auto HashVoxel = [](int x, int y, int z) -> uint64_t {
		return (static_cast<uint64_t>(x) & 0x1FFFFF) << 42 |
			(static_cast<uint64_t>(y) & 0x1FFFFF) << 21 |
			(static_cast<uint64_t>(z) & 0x1FFFFF);
		};

	vector<uint64_t> vertexVoxelKeys(numVertices, 0);

	voxelGridFilter.Reset();
	int goodVerticesCount = 0;

	// Apply calibration and remove points outside bounds
	for (unsigned int vertexIndex = 0; vertexIndex < numVertices; vertexIndex++)
	{
		Point3f temp = localVertices[vertexIndex];

		if (calibration.isCalibrated)
		{
			// Rotate the point to match the calibration
			temp.X += calibration.worldT[0];
			temp.Y += calibration.worldT[1];
			temp.Z += calibration.worldT[2];
			temp = RotatePoint(temp, calibration.worldR);

			// Remove the point if it is outside the bounds specified in the settings
			if (temp.X < bounds[0] || temp.X > bounds[3] ||
				temp.Y < bounds[1] || temp.Y > bounds[4] ||
				temp.Z < bounds[2] || temp.Z > bounds[5])
			{
				allVertices[vertexIndex] = invalidPoint;
				continue;
			}

			// Only keep the point if there is not already data for the same reduced point when considering the range
			else if (!voxelGridFilter.Insert(temp.X, temp.Y, temp.Z))
			{
				allVertices[vertexIndex] = invalidPoint;
				continue;
			}
		}

		allVertices[vertexIndex] = temp;
		goodVerticesCount++;

		// Compute this vertex's hash key for the density-based filter
		int vx = static_cast<int>(floor(temp.X / densityVoxelSize));
		int vy = static_cast<int>(floor(temp.Y / densityVoxelSize));
		int vz = static_cast<int>(floor(temp.Z / densityVoxelSize));
		uint64_t key = HashVoxel(vx, vy, vz);
		vertexVoxelKeys[vertexIndex] = key;
		voxelCounts[key]++;
	}

	const float downsampleVoxelSize = 0.0035f; // Slight downsample for meshing speed
	std::unordered_map<uint64_t, bool> downsampleOccupied;

	goodVertices.reserve(numVertices);
	goodColorPoints.reserve(numVertices);

	for (unsigned int i = 0; i < allVertices.size(); i++)
	{
		if (!allVertices[i].Invalid)
		{
			uint64_t densityKey = vertexVoxelKeys[i];

			if (voxelCounts[densityKey] < minPointsPerVoxel)
			{
				continue;
			}

			int vx = static_cast<int>(floor(allVertices[i].X / downsampleVoxelSize));
			int vy = static_cast<int>(floor(allVertices[i].Y / downsampleVoxelSize));
			int vz = static_cast<int>(floor(allVertices[i].Z / downsampleVoxelSize));
			uint64_t downsampleKey = HashVoxel(vx, vy, vz);

			if (!downsampleOccupied[downsampleKey])
			{
				downsampleOccupied[downsampleKey] = true;
				goodVertices.push_back(allVertices[i]);
				goodColorPoints.push_back(localColors[i]);
			}
		}
	}
}

bool LiveScanClient::ConstructMesh(std::vector<Point3f> &goodVertices, std::vector<int> &tempMeshIndices)
{
	// Clear data from the previous frame
	cloud->clear();
	normals->clear();
	cloudWithNormals->clear();
	mesh.polygons.clear();

	// Convert point cloud into a PCL point cloud (depth only)
	for (auto& p : goodVertices) cloud->push_back(pcl::PointXYZ(p.X, p.Y, p.Z));

	// Check that there are at least 3 points in the point cloud
	if (cloud->size() < 3)
	{
		std::lock_guard<std::mutex> lock(dataMutex);
		lastFrameVertices.clear();
		lastFrameColors.clear();
		lastFrameMeshIndices.clear();
		return false;
	}

	// Choose how many neighbours will be considered when calculating each point's surface normal (at most 50)
	size_t kSearch = std::min<size_t>(50, cloud->size());

	// Compute a normal vector for each point in the cloud
	ne.setInputCloud(cloud);
	ne.setKSearch(kSearch);
	ne.compute(*normals);

	for (size_t i = 0; i < cloud->size(); i++)
	{
		// Skip any point with invalid coordinates or components
		if (!pcl::isFinite(cloud->points[i])) return false;

		if (i >= normals->size() ||
			std::isnan(normals->points[i].normal_x) ||
			std::isnan(normals->points[i].normal_y) ||
			std::isnan(normals->points[i].normal_z))
		{
			return false;
		}

		// Create a PCL point containing both the point and its normal
		pcl::PointNormal pn;
		pn.x = cloud->points[i].x;
		pn.y = cloud->points[i].y;
		pn.z = cloud->points[i].z;
		pn.normal_x = normals->points[i].normal_x;
		pn.normal_y = normals->points[i].normal_y;
		pn.normal_z = normals->points[i].normal_z;

		cloudWithNormals->points.push_back(pn);
	}

	// Check that the resulting point cloud still has at least 3 valid points
	if (cloudWithNormals->size() < 3) return false;

	gp3.setInputCloud(cloudWithNormals);

	// Perform Greedy Projection Triangulation to obtain a mesh from the point cloud and its normals
	try { gp3.reconstruct(mesh); }
	catch (...) { Log("Mesh reconstruction failed."); }

	// Extract mesh vertices
	lastFrameMeshVertices.clear();
	pcl::PointCloud<pcl::PointXYZ> meshVerts;
	pcl::fromPCLPointCloud2(mesh.cloud, meshVerts);
	for (auto& v : meshVerts.points)
	{
		lastFrameMeshVertices.push_back(v.x);
		lastFrameMeshVertices.push_back(v.y);
		lastFrameMeshVertices.push_back(v.z);
	}

	// Extract mesh triangles
	for (const auto& poly : mesh.polygons)
	{
		if (poly.vertices.size() == 3)
		{
			tempMeshIndices.push_back(poly.vertices[0]);
			tempMeshIndices.push_back(poly.vertices[1]);
			tempMeshIndices.push_back(poly.vertices[2]);
		}
	}

	return true;
}

void LiveScanClient::ProcessDocument()
{
	if (captureManager->lastDocumentData.empty())
	{
		return;
	}

	cv::Mat newDocumentData = captureManager->lastDocumentData;
	float newDocumentScore = captureManager->lastDocumentScore;
	short newDocumentWidth = captureManager->lastDocumentWidth;
	short newDocumentHeight = captureManager->lastDocumentHeight;

	// Check if LiveScanClient has no data yet
	if (lastDocumentData.empty()) {
		lastDocumentData = newDocumentData;
		lastDocumentScore = newDocumentScore;
		lastDocumentWidth = newDocumentWidth;
		lastDocumentHeight = newDocumentHeight;
		isSendDocumentRequested = true;
		return;
	}

	float diff = ComputeImageDifference(newDocumentData);

	auto now = std::chrono::steady_clock::now();
	auto nowMs = std::chrono::time_point_cast<std::chrono::milliseconds>(now).time_since_epoch().count();

	if (nowMs - lastDocumentSendTime.count() >= DocumentSendTimeout || diff > DocumentDiffThreshold || newDocumentScore > lastDocumentScore)
	{
		lastDocumentData = newDocumentData;
		lastDocumentScore = newDocumentScore;
		lastDocumentWidth = newDocumentWidth;
		lastDocumentHeight = newDocumentHeight;
		isSendDocumentRequested = true;
		lastDocumentSendTime = std::chrono::milliseconds(nowMs);
	}
}

float LiveScanClient::ComputeImageDifference(cv::Mat& newDocumentData)
{
	if (newDocumentData.empty())
		return 1.0f; // Completely different if new is empty

	if (lastDocumentData.empty())
	{
		lastDocumentData = newDocumentData;
		return 1.0f; // No previous frame, assume max difference
	}

	// Resize to same dimensions
	cv::Mat resizedLast;
	cv::resize(lastDocumentData, resizedLast, newDocumentData.size());

	// Compute absolute difference
	cv::Mat diff;
	cv::absdiff(newDocumentData, resizedLast, diff);

	// Convert to grayscale to simplify metric
	cv::Mat grayDiff;
	cv::cvtColor(diff, grayDiff, cv::COLOR_BGR2GRAY);

	// Compute mean difference
	double meanDiff = cv::mean(grayDiff)[0]; // average intensity difference (0-255)

	// Normalize to 0.0-1.0
	float normalizedDiff = static_cast<float>(meanDiff / 255.0);

	// Update stored frame
	lastDocumentData = newDocumentData;

	return normalizedDiff;
}

void LiveScanClient::SendSerialNumber()
{
	if (wrapper && wrapper->sendSerialNumberCallback)
		wrapper->sendSerialNumberCallback(clientIndex, captureManager->serialNumber.c_str());
}

void LiveScanClient::ConfirmRecorded()
{
	if (wrapper && wrapper->confirmRecordedCallback)
		wrapper->confirmRecordedCallback(clientIndex);

	isConfirmRecordedRequested = false;
}

void LiveScanClient::ConfirmCalibrated()
{
	if (wrapper && wrapper->confirmCalibratedCallback)
	{
		float* R = new float[9] {
			calibration.worldR[0][0], calibration.worldR[0][1], calibration.worldR[0][2],
			calibration.worldR[1][0], calibration.worldR[1][1], calibration.worldR[1][2],
			calibration.worldR[2][0], calibration.worldR[2][1], calibration.worldR[2][2]
		};

		float* t = calibration.worldT.data();

		wrapper->confirmCalibratedCallback(clientIndex, calibration.usedMarkerId, R, t);
	}

	isConfirmCalibratedRequested = false;
}

void LiveScanClient::SendLatestFrame()
{
	if (wrapper && wrapper->sendLatestFrameCallback)
	{
		std::lock_guard<std::mutex> lock(dataMutex);

		int count = static_cast<int>(lastFrameVertices.size());
		if (count != lastFrameColors.size())
		{
			Log("[LiveScanClient] Warning: size mismatch! There were " + std::to_string(count) + " vertices and " + std::to_string(lastFrameColors.size()) + " colors. Sending smallest size.");

			if (count < lastFrameColors.size())
				count = lastFrameColors.size();
		}

		wrapper->sendLatestFrameCallback(clientIndex, lastFrameVertices.data(), lastFrameColors.data(), count);
	}


}

void LiveScanClient::SendLatestMesh()
{
	if (wrapper && wrapper->sendLatestMeshCallback)
	{
		std::lock_guard<std::mutex> lock(dataMutex);

		wrapper->sendLatestMeshCallback(
			clientIndex,
			lastFrameMeshIndices.data(),
			(int)lastFrameMeshIndices.size()
		);
	}
}

void LiveScanClient::SendRecordedFrame(std::vector<Point3s>& vertices, std::vector<RGB>& RGB, bool noMoreFrames)
{
	if (wrapper && wrapper->sendStoredFrameCallback)
	{
		int count = static_cast<int>(vertices.size());
		if (count != RGB.size())
		{
			Log("[LiveScanClient] Warning: size mismatch! There were " + std::to_string(count) + " vertices and " + std::to_string(RGB.size()) + " colors. Sending smallest size.");

			if (count < RGB.size())
				count = RGB.size();
		}

		wrapper->sendStoredFrameCallback(clientIndex, vertices.data(), RGB.data(), count, noMoreFrames);
	}
}

void LiveScanClient::ConfirmSyncState()
{
	if (wrapper && wrapper->confirmSyncStateCallback)
	{
		int syncStateToSend = 2; // default: Standalone
		switch (currentSyncState)
		{
		case Subordinate: 
			syncStateToSend = 0;
			break;
		case Master:      
			syncStateToSend = 1;
			break;
		case Standalone:  
			syncStateToSend = 2;
			break;
		}

		wrapper->confirmSyncStateCallback(clientIndex, syncStateToSend);
	}

	isConfirmSyncStateRequested = false;
}

void LiveScanClient::ConfirmMasterRestart()
{
	if (wrapper && wrapper->confirmMasterRestartCallback)
	{
		wrapper->confirmMasterRestartCallback(clientIndex);
	}

	isConfirmRestartAsMasterRequested = false;
}

void LiveScanClient::SendDocument()
{
	if (wrapper && wrapper->sendDocumentCallback)
	{
		wrapper->sendDocumentCallback(clientIndex, lastDocumentData.data, lastDocumentScore, lastDocumentWidth, lastDocumentHeight);
	}

	isSendDocumentRequested = false;
}

void LiveScanClient::OutputFrameToJson(std::vector<Point3s> vertices, std::vector<RGB> colors, std::vector<int> meshIndices)
{
	using json = nlohmann::json;
	json j;
	json verts = json::array();
	for (const auto& v : vertices) {
		verts.push_back({ {"x", v.X}, {"y", v.Y}, {"z", v.Z} });
	}
	j["vertices"] = verts;

	json cols = json::array();
	for (const auto& c : colors) {
		cols.push_back({ {"r", c.Red}, {"g", c.Green}, {"b", c.Blue} });
	}
	j["colors"] = cols;

	json tris = json::array();
	for (size_t i = 0; i < meshIndices.size(); ++i) tris.push_back(meshIndices[i]);
	j["triangles"] = tris;

	std::ofstream out("frame_cam" + std::to_string(clientIndex) + ".json");
	out << j.dump(4);
	out.close();
}

/// <summary>
/// Creates a log file for this particular instance of the LiveScanClient
/// </summary>
/// <param name="clientIndex">Index of the current client instance</param>
void LiveScanClient::SetupLogging(int clientIndex)
{
	wchar_t buffer[MAX_PATH];
	GetModuleFileNameW(NULL, buffer, MAX_PATH);
	std::wstring path(buffer);
	std::wstring dir = path.substr(0, path.find_last_of(L"\\/")) + L"\\Log";

	CreateDirectoryW(dir.c_str(), NULL);

	std::wstring logPath = dir + L"\\LiveScanClient_" + std::to_wstring(clientIndex) + L"_Log.txt";
	logFile.open(logPath, std::ios::out | std::ios::app);

	if (!logFile.is_open())
	{
		OutputDebugStringW(L"Failed to open log file.\n");
		return;
	}

	Log("==== Application Started (Client " + std::to_string(clientIndex) + ") ====");
}

/// <summary>
/// Returns a reference to the Log function. Can be used to pass this function to other modules of the 
/// LiveScanClient project to enable logging.
/// </summary>
/// <returns>A reference to the Log function</returns>
std::function<void(const std::string&)> LiveScanClient::GetLogger() {
	return [this](const std::string& msg) { this->Log(msg); };
}

/// <summary>
/// Appends the provided message to the end of the previously opened log file.
/// </summary>
/// <param name="message">Message to append to the log file</param>
void LiveScanClient::Log(const std::string& message)
{
	// Get current time
	auto now = std::chrono::system_clock::now();
	auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
	auto value = now_ms.time_since_epoch();
	long duration = value.count();
	std::time_t now_c = std::chrono::system_clock::to_time_t(now);

	std::tm tm;
	localtime_s(&tm, &now_c); // Windows-specific thread-safe function

	// Extract milliseconds
	int milliseconds = static_cast<int>(duration % 1000);

	// Format time to string with milliseconds
	std::ostringstream timestamp;
	timestamp << std::put_time(&tm, "[%Y-%m-%d %H:%M:%S")
		<< "." << std::setfill('0') << std::setw(3) << milliseconds
		<< "] " << message;

	std::string logEntry = timestamp.str();

	if (logFile.is_open())
	{
		logFile << logEntry << std::endl;
		logFile.flush();
	}
	else
	{
		OutputDebugStringA((logEntry + "\n").c_str()); // fallback to debugger output
	}
}

