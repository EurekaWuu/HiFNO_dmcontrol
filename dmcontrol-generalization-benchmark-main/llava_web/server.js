const express = require('express');
const http = require('http');
const path = require('path');
const net = require('net');
const bodyParser = require('body-parser');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;
const SOCKET_HOST = 'localhost';
const SOCKET_PORT = 7788;

// 中间件设置
app.use(express.static(path.join(__dirname)));
app.use(bodyParser.json({limit: '50mb'}));
app.use(bodyParser.urlencoded({limit: '50mb', extended: true}));

// 记录日志的辅助函数
function logMessage(message) {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] ${message}`);
    fs.appendFileSync('server.log', `[${timestamp}] ${message}\n`);
}

// API端点：处理聊天请求
app.post('/api/chat', (req, res) => {
    const payload = req.body;
    logMessage(`收到请求: ${JSON.stringify(payload).substring(0, 200)}...`);
    
    let retryCount = 0;
    const maxRetries = 3;
    
    function attemptSocketConnection() {
        const client = new net.Socket();
        let responseData = Buffer.from('');
        let connectTimeout;

        // 设置连接超时
        connectTimeout = setTimeout(() => {
            logMessage('连接超时');
            client.destroy();
            handleRetryOrFail(new Error('连接超时'));
        }, 5000);

        client.connect(SOCKET_PORT, SOCKET_HOST, () => {
            clearTimeout(connectTimeout);
            logMessage('已连接到Socket服务器');
            
            // 发送数据到Socket服务器
            client.write(JSON.stringify(payload));
            client.end();
            logMessage('数据已发送到Socket服务器');
        });

        // 接收数据
        client.on('data', (data) => {
            responseData = Buffer.concat([responseData, data]);
        });

        // 连接关闭
        client.on('close', () => {
            logMessage('Socket连接已关闭');
            try {
                const response = JSON.parse(responseData.toString());
                logMessage(`收到响应: ${JSON.stringify(response).substring(0, 200)}...`);
                res.json(response);
            } catch (e) {
                logMessage(`解析响应数据失败: ${e.message}`);
                logMessage(`原始数据长度: ${responseData.length}`);
                if (responseData.length > 0) {
                    logMessage(`原始数据前200个字符: ${responseData.toString().substring(0, 200)}`);
                }
                handleRetryOrFail(e);
            }
        });

        // 处理错误
        client.on('error', (err) => {
            logMessage(`Socket错误: ${err.message}`);
            client.destroy();
            handleRetryOrFail(err);
        });
    }

    function handleRetryOrFail(error) {
        if (retryCount < maxRetries) {
            retryCount++;
            logMessage(`尝试第${retryCount}次重新连接...`);
            setTimeout(attemptSocketConnection, 1000 * retryCount);
        } else {
            logMessage('达到最大重试次数，返回错误');
            res.status(500).json({
                error: `无法连接到服务器: ${error.message}`
            });
        }
    }

    // 开始第一次尝试
    attemptSocketConnection();
});

// 启动服务器
const server = http.createServer(app);
server.listen(PORT, () => {
    logMessage(`服务器运行在 http://localhost:${PORT}`);
});

process.on('uncaughtException', (err) => {
    logMessage(`未捕获的异常: ${err}`);
});

process.on('unhandledRejection', (reason, promise) => {
    logMessage(`未处理的Promise拒绝: ${reason}`);
}); 