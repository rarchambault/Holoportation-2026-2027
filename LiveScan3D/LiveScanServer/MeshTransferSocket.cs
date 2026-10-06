/***************************************************************************\

Module Name:  MeshTransferSocket.cs
Project:      LiveScan3D
Authors:      Roxanne Archambault
Copyright (c) Canadian Space Agency.

<Description>
This module is the socket used to send mesh data to connected clients.

This code was adapted from the following research: 
Kowalski, M.; Naruniec, J.; Daniluk, M.: "LiveScan3D: A Fast and Inexpensive 
3D Data Acquisition System for Multiple Kinect v2 Sensors". in 3D Vision (3DV), 
2015 International Conference on, Lyon, France, 2015

\***************************************************************************/

using System;
using System.Collections.Generic;
using System.Net.Sockets;

namespace LiveScanServer
{
    public class MeshTransferSocket : TransferSocketBase
    {
        private const float Range = 3.0f;
        private const float HalfRange = Range / 2.0f;
        
        private const float MinPrecision = Range / 65535f; // 16-BIT PRECISION
        private const float MaxScale = 1f / MinPrecision;

        private const float xRangeCenter = 0.0f;
        private const float yRangeCenter = 0.0f;
        private const float zRangeCenter = 1.0f; // Camera usually looks forward 1m

        public MeshTransferSocket(TcpClient clientSocket) : base(clientSocket) { }

        public void SendMesh(List<float> vertices, List<byte> colors, List<int> indices)
        {
            byte[] requestBuffer = Receive(1);

            while (requestBuffer.Length != 0)
            {
                if (requestBuffer[0] == 0)
                {
                    int vertexCount = vertices.Count / 3;
                    float scale = MaxScale; // Always use max 16-bit precision

                    // Gather vertex data and convert to short using the scale
                    byte[] vertexBuffer = new byte[vertexCount * 3 * sizeof(ushort)];

                    for (int i = 0; i < vertices.Count; i += 3)
                    {
                        ushort bx = EncodeFloatToUShort(vertices[i], xRangeCenter, scale);
                        ushort by = EncodeFloatToUShort(vertices[i + 1], yRangeCenter, scale);
                        ushort bz = EncodeFloatToUShort(vertices[i + 2], zRangeCenter, scale);

                        int byteIndex = (i / 3) * 6;
                        vertexBuffer[byteIndex] = (byte)(bx & 0xFF);
                        vertexBuffer[byteIndex + 1] = (byte)(bx >> 8);
                        vertexBuffer[byteIndex + 2] = (byte)(by & 0xFF);
                        vertexBuffer[byteIndex + 3] = (byte)(by >> 8);
                        vertexBuffer[byteIndex + 4] = (byte)(bz & 0xFF);
                        vertexBuffer[byteIndex + 5] = (byte)(bz >> 8);
                    }

                    // Gather mesh triangles
                    int numTrianglesToSend = indices.Count / 3;
                    byte[] indexBuffer = new byte[indices.Count * sizeof(int)];
                    Buffer.BlockCopy(indices.ToArray(), 0, indexBuffer, 0, indexBuffer.Length);

                    // Write to the stream
                    try
                    {
                        byte[] scaleBytes = BitConverter.GetBytes(scale);
                        socket.GetStream().Write(scaleBytes, 0, scaleBytes.Length);

                        WriteInt(vertexCount);
                        socket.GetStream().Write(vertexBuffer, 0, vertexBuffer.Length);
                        socket.GetStream().Write(colors.ToArray(), 0, colors.Count);

                        WriteInt(numTrianglesToSend);
                        socket.GetStream().Write(indexBuffer, 0, indexBuffer.Length);
                    }
                    catch (Exception ex)
                    {
                        Logger.Log("Error while sending mesh data: " + ex.Message);
                    }
                }

                // Receive a new request byte to make sure the receiver is ready to receive
                requestBuffer = Receive(1);
            }
        }

        private ushort EncodeFloatToUShort(float value, float rangeCenter, float scale)
        {
            float result = (value + HalfRange - rangeCenter) * scale;
            if (result < 0f) result = 0f;
            if (result > 65535f) result = 65535f;
            return (ushort)result;
        }
    }
}