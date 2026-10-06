/***************************************************************************\

Module Name:  TransferServer.cs
Project:      LiveScan3D
Authors:      Roxanne Archambault
Copyright (c) Canadian Space Agency.

<Description>
This module is the server used to listen for client connections through TCP
and to send them mesh data at a high frequency

This code was adapted from the following research: 
Kowalski, M.; Naruniec, J.; Daniluk, M.: "LiveScan3D: A Fast and Inexpensive 
3D Data Acquisition System for Multiple Kinect v2 Sensors". in 3D Vision (3DV), 
2015 International Conference on, Lyon, France, 2015

\***************************************************************************/

using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;

namespace LiveScanServer
{
    public class TransferServer
    {
        public List<float> Vertices = new List<float>();
        public List<byte> Colors = new List<byte>();
        public List<int> MeshIndices = new List<int>();
        public DocumentInfo DocumentInfo = new DocumentInfo();

        private const int MeshPort = 48002;
        private const int DocumentPort = 48003;
        private const int CheckConnectionInterval = 1000;

        private TcpListener meshListener;
        private System.Timers.Timer meshConnectionTimer;
        private CancellationTokenSource meshCancellationTokenSource;
        private List<MeshTransferSocket> meshClients = new List<MeshTransferSocket>();
        private object meshClientLock = new object();
        private bool isMeshServerRunning = false;

        private TcpListener documentListener;
        private System.Timers.Timer documentConnectionTimer;
        private CancellationTokenSource documentCancellationTokenSource;
        private List<DocumentTransferSocket> documentClients = new List<DocumentTransferSocket>();
        private object documentClientLock = new object();
        private bool isDocumentServerRunning = false;

        ~TransferServer()
        {
            StopMeshServer();
            StopDocumentServer();
        }

        /// <summary>
        /// Starts the TCP listener and Tasks for the mesh server to listen for client connections and send them data
        /// </summary>
        public void StartMeshServer()
        {
            if (!isMeshServerRunning)
            {
                // Start TCP listener server
                meshListener = new TcpListener(IPAddress.Any, MeshPort);
                meshListener.Start();

                isMeshServerRunning = true;

                // Start tasks to listen for client connections and send data
                meshCancellationTokenSource = new CancellationTokenSource();
                Task.Run(() => ConnectMeshClients(meshCancellationTokenSource.Token));
                Task.Run(() => SendMeshToAllClients(meshCancellationTokenSource.Token));

                // Start a timer to ping connected clients at a regular interval to ensure they are still connected
                meshConnectionTimer = new System.Timers.Timer();
                meshConnectionTimer.Interval = CheckConnectionInterval;

                meshConnectionTimer.Elapsed += delegate (object sender, System.Timers.ElapsedEventArgs e)
                {
                    lock (meshClientLock)
                    {
                        for (int i = 0; i < meshClients.Count; i++)
                        {
                            if (!meshClients[i].IsConnected())
                            {
                                meshClients.RemoveAt(i);
                                i--;
                            }
                        }
                    }
                };

                meshConnectionTimer.Start();
            }
        }

        /// <summary>
        /// Starts the TCP listener and Tasks for the document server to listen for client connections and send them data
        /// </summary>
        public void StartDocumentServer()
        {
            if (!isDocumentServerRunning)
            {
                // Start TCP listener server
                documentListener = new TcpListener(IPAddress.Any, DocumentPort);
                documentListener.Start();

                isDocumentServerRunning = true;

                // Start tasks to listen for client connections and send data
                documentCancellationTokenSource = new CancellationTokenSource();
                Task.Run(() => ConnectDocumentClients(documentCancellationTokenSource.Token));
                Task.Run(() => SendDocumentToAllClients(documentCancellationTokenSource.Token));

                // Start a timer to ping connected clients at a regular interval to ensure they are still connected
                documentConnectionTimer = new System.Timers.Timer();
                documentConnectionTimer.Interval = CheckConnectionInterval;

                documentConnectionTimer.Elapsed += delegate (object sender, System.Timers.ElapsedEventArgs e)
                {
                    lock (documentClientLock)
                    {
                        for (int i = 0; i < documentClients.Count; i++)
                        {
                            if (!documentClients[i].IsConnected())
                            {
                                documentClients.RemoveAt(i);
                                i--;
                            }
                        }
                    }
                };

                documentConnectionTimer.Start();
            }
        }

        /// <summary>
        /// Stops the mesh server and all associated threads and connections
        /// </summary>
        public void StopMeshServer()
        {
            if (isMeshServerRunning)
            {
                isMeshServerRunning = false;

                // Stop checking client connections
                meshConnectionTimer.Stop();

                // Explicitly stop Tasks
                meshCancellationTokenSource.Cancel();

                // Stop each client socket and its threads
                foreach (MeshTransferSocket clientSocket in meshClients)
                {
                    clientSocket.Stop();
                }

                // Stop the listener server
                meshListener.Stop();

                lock (meshClientLock)
                    meshClients.Clear();
            }
        }

        /// <summary>
        /// Stops the document server and all associated threads and connections
        /// </summary>
        public void StopDocumentServer()
        {
            if (isDocumentServerRunning)
            {
                isDocumentServerRunning = false;

                // Stop checking client connections
                documentConnectionTimer.Stop();

                // Explicitly stop Tasks
                documentCancellationTokenSource.Cancel();

                // Stop each client socket and its threads
                foreach (DocumentTransferSocket clientSocket in documentClients)
                {
                    clientSocket.Stop();
                }

                // Stop the listener server
                documentListener.Stop();

                lock (documentClientLock)
                    documentClients.Clear();
            }
        }

        /// <summary>
        /// Listens for mesh client connections in a loop
        /// </summary>
        /// <param name="token">Cancellation token to stop the task</param>
        /// <returns>Task representing the listener</returns>
        private async Task ConnectMeshClients(CancellationToken token)
        {
            while (isMeshServerRunning && !token.IsCancellationRequested)
            {
                try
                {
                    // Try to accept a new client
                    TcpClient newClient = meshListener.AcceptTcpClient();

                    // Add the new client to the list
                    lock (meshClientLock)
                    {
                        meshClients.Add(new MeshTransferSocket(newClient));
                    }
                }
                catch (SocketException)
                {
                }

                await Task.Delay(100);
            }
        }

        /// <summary>
        /// Listens for document client connections in a loop
        /// </summary>
        /// <param name="token">Cancellation token to stop the task</param>
        /// <returns>Task representing the listener</returns>
        private async Task ConnectDocumentClients(CancellationToken token)
        {
            while (isDocumentServerRunning && !token.IsCancellationRequested)
            {
                try
                {
                    // Try to accept a new client
                    TcpClient newClient = documentListener.AcceptTcpClient();

                    // Add the new client to the list
                    lock (documentClientLock)
                    {
                        documentClients.Add(new DocumentTransferSocket(newClient));
                    }
                }
                catch (SocketException)
                {
                }

                await Task.Delay(100);
            }
        }

        /// <summary>
        /// Sends mesh data to all connected clients at regular intervals
        /// </summary>
        /// <param name="token">Cancellation token to stop the Task</param>
        /// <returns>Task representing the sender</returns>
        private async Task SendMeshToAllClients(CancellationToken token)
        {
            while (isMeshServerRunning && !token.IsCancellationRequested)
            {
                // Send latest mesh to all connected clients
                for (int i = 0; i < meshClients.Count; i++)
                {
                    // Send a mesh frame
                    lock (Vertices)
                    {
                        meshClients[i].SendMesh(Vertices, Colors, MeshIndices);
                    }
                }

                await Task.Delay(10);
            }
        }

        /// <summary>
        /// Sends document data to all connected clients at regular intervals
        /// </summary>
        /// <param name="token">Cancellation token to stop the Task</param>
        /// <returns>Task representing the sender</returns>
        private async Task SendDocumentToAllClients(CancellationToken token)
        {
            while (isDocumentServerRunning && !token.IsCancellationRequested)
            {
                if (DocumentInfo.IsNew)
                {
                    lock (DocumentInfo)
                    {
                        // Send latest document to all connected clients
                        for (int i = 0; i < documentClients.Count; i++)
                        {
                            documentClients[i].SendDocument(DocumentInfo.Data, DocumentInfo.Width, DocumentInfo.Height);
                        }
                    }

                    DocumentInfo.IsNew = false;
                }

                await Task.Delay(100);
            }
        }
    }
}
