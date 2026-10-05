Shader "Custom/VertexColorMesh"
{
    SubShader
    {
        Tags
        {
            "RenderType" = "Opaque"
            "Queue" = "Geometry"
        }

        Pass
        {
            Cull Back
            ZWrite On
            ZTest LEqual

            CGPROGRAM

            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct VertexInput
            {
                float3 position : POSITION;
                float4 color : COLOR;
                float3 normal : NORMAL;
            };

            struct VertexOutput
            {
                float4 position : SV_POSITION;
                float4 color : COLOR;
            };

            VertexOutput vert(VertexInput input)
            {
                VertexOutput output;

                output.position = UnityObjectToClipPos(input.position);
                output.color = input.color;

                return output;
            }

            half4 frag(VertexOutput input) : SV_Target
            {
                return input.color;
            }

            ENDCG
        }
    }
}