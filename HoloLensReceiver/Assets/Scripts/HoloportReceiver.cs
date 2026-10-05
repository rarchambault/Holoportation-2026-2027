/***************************************************************************\

Module Name:  HoloportReceiver.cs
Project:      HoloLensReceiver
Authors:      Roxanne Archambault
Copyright (c) Canadian Space Agency.

<Description>
This module receives meshes and documents from a TCP server and sends 
them to the appropriate renderers.

This code was adapted from the following research: 
Kowalski, M.; Naruniec, J.; Daniluk, M.: "LiveScan3D: A Fast and Inexpensive 
3D Data Acquisition System for Multiple Kinect v2 Sensors". in 3D Vision (3DV), 
2015 International Conference on, Lyon, France, 2015

\***************************************************************************/

using System;
using System.Net.Sockets;
using System.Threading.Tasks;
using UnityEngine;

public class HoloportReceiver : MonoBehaviour
{
    public string ServerIPAddress = "127.0.0.1";
    public bool IsServerIPAddressSet = false;
    public int PointCloudPort = 48002;
    public int DocumentPort = 48003;
    public float ConnectionRetryInterval = 10.0f;

    // Parameters used to deserialize point clouds
    private const float Range = 3.0f;
    private const float HalfRange = Range / 2.0f;
    private const float xRangeCenter = 0.0f;
    private const float yRangeCenter = 0.0f;
    private const float zRangeCenter = 1.0f;

    private TcpClient pointCloudClient;
    private bool isPointCloudClientConnected = false;
    private bool isPointCloudClientConnecting = false;
    private float pointCloudConnectionTimer = 0.0f;

    private TcpClient documentClient;
    private bool isDocumentClientConnected = false;
    private bool isDocumentClientConnecting = false;
    private float documentConnectionTimer = 0.0f;
    
    private StreamingMeshRenderer streamingMeshRenderer;
    private DocumentRenderer documentRenderer;

    private MeshRenderer meshRenderer;
        
    private void Start()
    {
        streamingMeshRenderer = GetComponent<StreamingMeshRenderer>();
        documentRenderer = GetComponent<DocumentRenderer>();
        meshRenderer = GetComponent<MeshRenderer>();
    }

    private void Update()
    {
        if (!isPointCloudClientConnecting && IsServerIPAddressSet)
        {
            isPointCloudClientConnecting = true;
            ConnectPointCloudClient();
        }

        if (!isDocumentClientConnecting && IsServerIPAddressSet)
        {
            isDocumentClientConnecting = true;
            ConnectDocumentClient();
        }

        if (isPointCloudClientConnecting && !isPointCloudClientConnected)
        {
            pointCloudConnectionTimer += Time.deltaTime;

            if (pointCloudConnectionTimer >= ConnectionRetryInterval)
            {
                // Retry connecting at regular intervals if connection failed
                ConnectPointCloudClient();
                pointCloudConnectionTimer = 0.0f;
            }
        }

        if (isDocumentClientConnecting && !isDocumentClientConnected)
        {
            documentConnectionTimer += Time.deltaTime;

            if (documentConnectionTimer >= ConnectionRetryInterval)
            {
                // Retry connecting at regular intervals if connection failed
                ConnectDocumentClient();
                documentConnectionTimer = 0.0f;
            }
        }
    }

    private async void ConnectPointCloudClient()
    {
        pointCloudClient = new TcpClient();

        try
        {
            await pointCloudClient.ConnectAsync(ServerIPAddress, PointCloudPort);
            isPointCloudClientConnected = true;
            Debug.Log("Connected to LiveScan3D point cloud server.");
            ReceivePointClouds();
            meshRenderer.enabled = true;
        }
        catch (Exception e)
        {
            Debug.LogError("Connection to LiveScan3D point cloud server failed: " + e.Message);
            isPointCloudClientConnected = false;
            isPointCloudClientConnecting = false;
        }
    }

    private async void ConnectDocumentClient()
    {
        documentClient = new TcpClient();

        try
        {
            await documentClient.ConnectAsync(ServerIPAddress, DocumentPort);
            isDocumentClientConnected = true;
            Debug.Log("Connected to LiveScan3D document server.");
            ReceiveDocuments();
        }
        catch (Exception e)
        {
            Debug.LogError("Connection to LiveScan3D document server failed: " + e.Message);
            isDocumentClientConnected = false;
            isDocumentClientConnecting = false;
        }
    }

    private async void ReceivePointClouds()
    {
        while (isPointCloudClientConnected && pointCloudClient != null && pointCloudClient.Connected)
        {
            try
            {
                // Request a new frame
                await pointCloudClient.GetStream().WriteAsync(new byte[] { 0 }, 0, 1);

                // Read scale factor (short)
                float scale = await ReadFloatAsync(pointCloudClient);

                // Read number of vertices (int)
                int vertexCount = await ReadIntAsync(pointCloudClient);

                // Initialize arrays for vertices and colors data
                int vertexByteCount = vertexCount * 3 * sizeof(ushort);
                int colorByteCount = vertexCount * 3;

                // Read vertices and colors data
                byte[] verticesBytes = await ReadAsync(pointCloudClient, vertexByteCount);
                byte[] colorsBytes = await ReadAsync(pointCloudClient, colorByteCount);

                // Read number of triangles (int)
                int triangleCount = await ReadIntAsync(pointCloudClient);
                int indexCount = triangleCount * 3;

                // Read triangle index data
                int indicesByteCount = indexCount * sizeof(int);
                byte[] indicesBytes = await ReadAsync(pointCloudClient, indicesByteCount);

                int[] meshIndices = new int[indexCount];
                Buffer.BlockCopy(indicesBytes, 0, meshIndices, 0, indicesByteCount);

                Vector3[] vertices;
                Color32[] colors;

                DeserializePointCloud(vertexCount, scale, verticesBytes, colorsBytes, out vertices, out colors);
                streamingMeshRenderer.EnqueueMesh(vertices, colors, meshIndices);
            }
            catch (Exception)
            {
                if (pointCloudClient != null && !pointCloudClient.Connected && isPointCloudClientConnected)
                {
                    // The socket was disconnected while trying to receive a point cloud; close the socket and hide the renderer
                    isPointCloudClientConnecting = false;
                    isPointCloudClientConnected = false;
                    try { pointCloudClient.Close(); pointCloudClient.Dispose(); } catch { }
                    meshRenderer.enabled = false;
                }
            }
        }
    }

    private async void ReceiveDocuments()
    {
        while (isDocumentClientConnected && documentClient != null && documentClient.Connected)
        {
            try
            {
                // Read width and height
                short width = await ReadShortAsync(documentClient);
                short height = await ReadShortAsync(documentClient);
                int dataSize = await ReadIntAsync(documentClient);
                byte[] dataBytes = await ReadAsync(documentClient, dataSize);

                documentRenderer.EnqueueDocument(width, height, dataBytes);
            }
            catch (Exception e)
            {
                if (documentClient != null && !documentClient.Connected && isDocumentClientConnected)
                {
                    // The socket was disconnected while trying to receive a point cloud; close the socket
                    isDocumentClientConnecting = false;
                    isDocumentClientConnected = false;
                    try { documentClient.Close(); documentClient.Dispose(); } catch { }
                }
            }
        }
    }

    private void DeserializePointCloud(int numPoints, float scale, byte[] verticesBytes, byte[] colorsBytes, out Vector3[] vertices, out Color32[] colors)
    {
        vertices = new Vector3[numPoints];
        colors = new Color32[numPoints];

        // Deserialize position data
        for (int i = 0; i < numPoints; i++)
        {
            int b = i * 6;
            ushort ux = BitConverter.ToUInt16(verticesBytes, b);
            ushort uy = BitConverter.ToUInt16(verticesBytes, b + 2);
            ushort uz = BitConverter.ToUInt16(verticesBytes, b + 4);

            float x = DecodeUShortToFloat(ux, xRangeCenter, scale);
            float y = DecodeUShortToFloat(uy, yRangeCenter, scale);
            float z = DecodeUShortToFloat(uz, zRangeCenter, scale);

            vertices[i] = new Vector3(x, y, z);
        }

        // Deserialize color data
        for (int i = 0; i < numPoints; i++)
        {
            int b = i * 3;
            colors[i] = new Color32(colorsBytes[b], colorsBytes[b + 1], colorsBytes[b + 2], 255);
        }
    }

    private float DecodeUShortToFloat(ushort val, float rangeCenter, float scale)
    {
        return (val / scale) - HalfRange + rangeCenter;
    }

    private async Task<short> ReadShortAsync(TcpClient client)
    {
        byte[] buffer = await ReadAsync(client, sizeof(short));
        return BitConverter.ToInt16(buffer, 0);
    }

    private async Task<int> ReadIntAsync(TcpClient client)
    {
        byte[] buffer = await ReadAsync(client, sizeof(int));
        return BitConverter.ToInt32(buffer, 0);
    }

    private async Task<float> ReadFloatAsync(TcpClient client)
    {
        byte[] buffer = await ReadAsync(client, sizeof(float));
        return BitConverter.ToSingle(buffer, 0);
    }

    private async Task<byte[]> ReadAsync(TcpClient client, int numBytesToRead)
    {
        byte[] buffer = new byte[numBytesToRead];
        int numBytesRead = 0;
        NetworkStream stream = client.GetStream();

        while (numBytesRead < numBytesToRead)
        {
            int read = await stream.ReadAsync(buffer, numBytesRead, numBytesToRead - numBytesRead);
            if (read == 0) throw new Exception("Socket closed while reading.");
            numBytesRead += read;
        }
        return buffer;
    }

    private void OnDestroy()
    {
        isPointCloudClientConnecting = false;
        isPointCloudClientConnected = false;

        if (pointCloudClient != null)
        { 
            try
            { 
                pointCloudClient.Close(); 
                pointCloudClient.Dispose(); 
            } 
            catch { } 
            
            pointCloudClient = null;
        }

        isDocumentClientConnecting = false;
        isDocumentClientConnected = false;

        if (documentClient != null) 
        { 
            try 
            { 
                documentClient.Close();
                documentClient.Dispose();
            } 
            catch { } 
            
            documentClient = null; }
    }
}