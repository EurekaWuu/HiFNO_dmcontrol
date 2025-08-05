import http.server
import socketserver
import json
import socket
import base64
from urllib.parse import urlparse, parse_qs
import os
import sys
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("xingling_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("XingLing-Server")

# 服务器设置
PORT = 8000
SOCKET_HOST = 'localhost'  # LLaVA socket服务器地址
SOCKET_PORT = 7788         # LLaVA socket服务器端口

class XingLingHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        logger.info("%s - %s" % (self.client_address[0], format % args))
        
    def do_POST(self):
        if self.path == '/api/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request = json.loads(post_data.decode('utf-8'))
            
            logger.info(f"收到API请求: {self.path}, 数据长度: {content_length}")
            
            # 连接到LLaVA Socket服务器
            try:
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.settimeout(60)  # 设置超时时间为60秒
                logger.info(f"尝试连接LLaVA服务器: {SOCKET_HOST}:{SOCKET_PORT}")
                client.connect((SOCKET_HOST, SOCKET_PORT))
                logger.info("已连接到LLaVA服务器，发送数据...")
                client.sendall(json.dumps(request).encode())
                client.shutdown(socket.SHUT_WR)
                
                # 接收来自LLaVA服务器的响应
                response_data = b''
                while True:
                    try:
                        data = client.recv(4096)
                        if not data:
                            break
                        response_data += data
                    except socket.timeout:
                        logger.error("接收数据超时")
                        break
                    
                client.close()
                
                # 如果响应为空，返回错误
                if not response_data:
                    logger.error("从LLaVA服务器收到空响应")
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    error_response = json.dumps({"error": "从模型服务器收到空响应"})
                    self.wfile.write(error_response.encode())
                    return
                
                # 将响应发送回客户端
                logger.info(f"收到LLaVA响应，长度: {len(response_data)}字节")
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')  # 允许跨域
                self.end_headers()
                self.wfile.write(response_data)
                
            except Exception as e:
                logger.error(f"处理请求时发生错误: {e}", exc_info=True)
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                error_response = json.dumps({"error": f"服务器错误: {str(e)}"})
                self.wfile.write(error_response.encode())
        else:
            # 对其它请求使用默认的处理方法（提供静态文件）
            return http.server.SimpleHTTPRequestHandler.do_POST(self)

    def do_OPTIONS(self):
        # 处理OPTIONS请求（支持CORS）
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def run_server():
    try:
        handler = XingLingHandler
        httpd = socketserver.TCPServer(("", PORT), handler)
        current_dir = os.getcwd()
        logger.info(f"服务器运行在 http://localhost:{PORT}")
        logger.info(f"提供目录: {current_dir}")
        logger.info(f"准备连接LLaVA服务器: {SOCKET_HOST}:{SOCKET_PORT}")
        logger.info("按Ctrl+C停止服务器")
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("服务器已停止")
        httpd.server_close()
        sys.exit(0)
    except Exception as e:
        logger.error(f"启动服务器时发生错误: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_server()