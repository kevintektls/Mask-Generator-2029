using System;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using System.Collections;

public class RaycastClient : MonoBehaviour
{
    [Header("Config")]
    public string host = "localhost";
    public int    port = 9999;
    public Camera trackCamera;

    private TcpClient   _client;
    private NetworkStream _stream;
    public  float[]     Distances { get; private set; }
    public  float[]     Angles    { get; private set; }

    void Start()
    {
        Connect();
        StartCoroutine(SendFrameLoop());
    }

    void Connect()
    {
        try {
            _client = new TcpClient(host, port);
            _stream = _client.GetStream();
            Debug.Log("Connecté au pipeline Python");
        } catch (Exception e) {
            Debug.LogError($"Connexion échouée : {e.Message}");
        }
    }

    IEnumerator SendFrameLoop()
    {
        while (true)
        {
            yield return new WaitForEndOfFrame();

            // Capture la caméra en JPEG
            RenderTexture rt = new RenderTexture(256, 256, 24);
            trackCamera.targetTexture = rt;
            trackCamera.Render();

            RenderTexture.active = rt;
            Texture2D tex = new Texture2D(256, 256, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, 256, 256), 0, 0);
            tex.Apply();

            trackCamera.targetTexture = null;
            RenderTexture.active = null;
            Destroy(rt);

            byte[] jpeg = tex.EncodeToJPG(85);
            Destroy(tex);

            // Envoie au pipeline Python
            try {
                SendBytes(jpeg);
                string json = ReceiveString();
                ParseResponse(json);
            } catch (Exception e) {
                Debug.LogWarning($"Erreur pipeline : {e.Message}");
                Connect(); // reconnexion automatique
            }
        }
    }

    void SendBytes(byte[] data)
    {
        // Envoie taille (4 bytes big-endian) puis données
        byte[] sizeBytes = BitConverter.GetBytes((uint)data.Length);
        if (BitConverter.IsLittleEndian) Array.Reverse(sizeBytes);
        _stream.Write(sizeBytes, 0, 4);
        _stream.Write(data, 0, data.Length);
    }

    string ReceiveString()
    {
        byte[] sizeBytes = new byte[4];
        _stream.Read(sizeBytes, 0, 4);
        if (BitConverter.IsLittleEndian) Array.Reverse(sizeBytes);
        int size = (int)BitConverter.ToUInt32(sizeBytes, 0);

        byte[] data = new byte[size];
        int received = 0;
        while (received < size)
            received += _stream.Read(data, received, size - received);

        return Encoding.UTF8.GetString(data);
    }

    void ParseResponse(string json)
    {
        // Parse manuel (évite une dépendance externe)
        // Pour un vrai projet, utilise Newtonsoft.Json
        var response = JsonUtility.FromJson<RaycastResponse>(json);
        Distances = response.distances;
        Angles    = response.angles_deg;
    }

    void OnDestroy()
    {
        _stream?.Close();
        _client?.Close();
    }

    [Serializable]
    class RaycastResponse
    {
        public float[] distances;
        public float[] angles_deg;
        public int     num_rays;
        public float   fov_deg;
        public float   inference_ms;
    }
}
