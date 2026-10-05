
/***************************************************************************\

Module Name:  StreamingMeshRenderer.cs
Project:      HoloLensReceiver
Authors:      Roxanne Archambault
Copyright (c) Canadian Space Agency.

<Description>
This module receives mesh data from the HoloportReceiver, enqueues them
and renders them.

This code was adapted from the following research: 
Kowalski, M.; Naruniec, J.; Daniluk, M.: "LiveScan3D: A Fast and Inexpensive 
3D Data Acquisition System for Multiple Kinect v2 Sensors". in 3D Vision (3DV), 
2015 International Conference on, Lyon, France, 2015

\***************************************************************************/

using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class StreamingMeshRenderer : MonoBehaviour
{
    [Tooltip("If true, turns off backface culling to prevent holes in the mesh.")]
    public bool doubleSided = true;

    [Tooltip("MUST be checked if using a Lit/Standard shader so light bounces correctly!")]
    public bool calculateNormals = true;

    public bool debug;

    private Mesh mesh;
    private MeshFilter meshFilter;
    private MeshRenderer meshRenderer;

    // Parameters used to calculate and log FPS
    private bool isStarted = false;
    private float timeSinceLastRender = 0.0f;
    private float totalTime = 0.0f;
    private int numFrames = 0;

    private void Awake()
    {
        meshFilter = GetComponent<MeshFilter>();
        meshRenderer = GetComponent<MeshRenderer>();

        mesh = new Mesh();
        mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        mesh.MarkDynamic();
        meshFilter.mesh = mesh;

        if (meshRenderer.sharedMaterial == null)
        {
            Shader shader = Shader.Find("Standard");

            if (shader != null)
            {
                meshRenderer.material = new Material(shader);
            }
        }

        if (doubleSided && meshRenderer.material.HasProperty("_Cull"))
        {
            meshRenderer.material.SetFloat("_Cull", (float)UnityEngine.Rendering.CullMode.Off);
        }
    }

    void Update()
    {
        if (isStarted)
        {
            timeSinceLastRender += Time.deltaTime;
        }
    }

    public void EnqueueMesh(Vector3[] vertices, Color32[] colors, int[] triangles)
    {
        isStarted = true;

        mesh.Clear(false);
        mesh.SetVertices(vertices);
        mesh.SetColors(colors);
        mesh.SetTriangles(triangles, 0);
        mesh.RecalculateBounds();

        if (calculateNormals) mesh.RecalculateNormals();

        // Calculate and log FPS
        totalTime += timeSinceLastRender;
        timeSinceLastRender = 0.0f;
        numFrames++;

        if (debug)
        {
            Debug.Log("Average FPS: " + numFrames / totalTime);
        }
    }
}