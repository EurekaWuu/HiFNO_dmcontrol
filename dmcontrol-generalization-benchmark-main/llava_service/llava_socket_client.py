import socket
import base64
import json

img_path = "/mnt/lustre/GPU4/home/wuhanpeng/LLaVA/images/llava_logo.png"
with open(img_path, "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "prompt": "Describe what you see in this image, focusing on the robot's posture, leg positions, and walking motion. Be specific and detailed.",
    "image_base64": img_base64
}

HOST, PORT = 'localhost', 7788
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    print("[Client] 连接服务器")
    s.connect((HOST, PORT))

    s.sendall(json.dumps(payload).encode())
    print("[Client] 请求已发送")
    s.shutdown(socket.SHUT_WR)  

    response = b''
    while True:
        packet = s.recv(4096)
        if not packet:
            break
        response += packet
    print("[Client] 收到响应:")
    print(response.decode())

    result = json.loads(response.decode())
    print("[Client] 解析后的结果:")
    print(result)


"""
python llava_socket_client.py


"""