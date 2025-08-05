document.addEventListener('DOMContentLoaded', function() {
    // 获取DOM元素
    const chatHistory = document.getElementById('chatHistory');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const clearChatBtn = document.getElementById('clearChat');
    const uploadImageBtn = document.getElementById('uploadImageBtn');
    const imageUpload = document.getElementById('imageUpload');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = imagePreview.querySelector('img');
    const removeImageBtn = document.getElementById('removeImage');
    const constitutionTestBtn = document.getElementById('constitutionTest');
    const featureBtns = document.querySelectorAll('.feature-btn');

    // 全局变量
    let currentImageBase64 = null;
    let currentFeature = 'general'; // 默认为一般健康咨询
    
    // 功能按钮点击事件
    featureBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const feature = this.getAttribute('data-feature');
            currentFeature = feature;
            
            // 更新所有按钮样式
            featureBtns.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            
            // 根据不同功能添加系统提示信息
            let systemMessage;
            switch(feature) {
                case 'tcm':
                    systemMessage = "已切换至中医药问答模式。您可以咨询有关中药、针灸、推拿、中医理论等相关问题。";
                    break;
                case 'weight':
                    systemMessage = "已切换至体重管理模式。您可以咨询减重、控制饮食、运动方案等相关问题，也可以上传身体状况图片进行分析。";
                    break;
                case 'chronic':
                    systemMessage = "已切换至慢病管理模式。您可以咨询高血压、糖尿病、高血脂等慢性病的管理方法和注意事项。";
                    break;
                case 'guide':
                    systemMessage = "已切换至智能导诊模式。请描述您的症状，系统会为您推荐合适的科室和就医建议。";
                    break;
                default:
                    systemMessage = "已切换至健康咨询模式。您可以咨询一般性健康问题，我会尽力解答。";
            }
            addMessage(systemMessage, 'system');
        });
    });

    // 发送按钮点击事件
    sendBtn.addEventListener('click', sendMessage);

    // 输入框回车事件
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // 清空聊天记录按钮事件
    clearChatBtn.addEventListener('click', function() {
        const confirmClear = confirm('确定要清空所有聊天记录吗？');
        if (confirmClear) {
            // 仅保留初始系统消息
            chatHistory.innerHTML = `
                <div class="message system-message">
                    <div class="message-content">
                        您好！我是"杏林千问"智能健康助手。请问有什么健康问题需要咨询吗？我可以回答健康咨询、中医药知识、体重管理等问题。
                    </div>
                </div>
            `;
            currentImageBase64 = null;
            imagePreview.classList.add('d-none');
        }
    });

    // 上传图片按钮事件
    uploadImageBtn.addEventListener('click', function() {
        imageUpload.click();
    });

    // 图片上传处理
    imageUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const base64Image = e.target.result;
                currentImageBase64 = base64Image.split(',')[1]; // 移除 "data:image/jpeg;base64," 部分
                previewImg.src = base64Image;
                imagePreview.classList.remove('d-none');
            };
            reader.readAsDataURL(file);
        }
    });

    // 移除图片按钮事件
    removeImageBtn.addEventListener('click', function() {
        currentImageBase64 = null;
        imagePreview.classList.add('d-none');
        imageUpload.value = '';
    });

    // 体质测试按钮事件
    constitutionTestBtn.addEventListener('click', function() {
        addMessage("中医体质辨识在线测试将帮助您了解自己的体质类型。请根据实际情况回答以下问题（是/否）：\n\n1. 您平时容易疲劳吗？\n2. 您的手脚经常发凉吗？\n3. 您容易出汗吗？\n4. 您容易感到口渴吗？\n5. 您的舌苔厚腻吗？", 'system');
    });

    // 发送消息到服务器
    function sendMessage() {
        const message = userInput.value.trim();
        if (!message && !currentImageBase64) {
            alert('请输入问题或上传图片！');
            return;
        }

        // 添加用户消息到聊天区
        if (message) {
            addMessage(message, 'user');
        } else if (currentImageBase64) {
            addMessage("已上传图片进行分析", 'user');
        }
        
        // 清空输入框
        userInput.value = '';
        
        // 显示系统正在思考的消息
        const thinkingMsgId = 'thinking-' + Date.now();
        addThinkingMessage(thinkingMsgId);
        
        // 准备发送的数据
        const payload = {
            prompt: message || "请分析这张图片",
            feature: currentFeature
        };
        
        if (currentImageBase64) {
            payload.image_base64 = currentImageBase64;
            // 发送后清除图片
            currentImageBase64 = null;
            imagePreview.classList.add('d-none');
        }
        
        // 发送数据到Socket服务器
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        })
        .then(response => response.json())
        .then(data => {
            // 移除思考中的消息
            removeThinkingMessage(thinkingMsgId);
            
            // 添加服务器响应到聊天区
            if (data.error) {
                addMessage("抱歉，处理请求时发生错误: " + data.error, 'system');
            } else {
                addMessage(data.description || data.response || "抱歉，未能获取有效回复", 'system');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            removeThinkingMessage(thinkingMsgId);
            addMessage("抱歉，连接服务器时发生错误，请稍后再试。", 'system');
        });
    }

    // 添加消息到聊天历史
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = text;
        
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        const now = new Date();
        timeDiv.textContent = now.getHours().toString().padStart(2, '0') + ':' + 
                             now.getMinutes().toString().padStart(2, '0');
        
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timeDiv);
        chatHistory.appendChild(messageDiv);
        
        // 滚动到底部
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    // 显示思考中的消息
    function addThinkingMessage(id) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message system-message thinking';
        messageDiv.id = id;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = '<span class="thinking-dots"><span>.</span><span>.</span><span>.</span></span> 正在思考中';
        
        messageDiv.appendChild(contentDiv);
        chatHistory.appendChild(messageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        
        // 添加思考动画
        animateThinkingDots();
    }
    
    // 移除思考中的消息
    function removeThinkingMessage(id) {
        const thinkingMsg = document.getElementById(id);
        if (thinkingMsg) {
            thinkingMsg.remove();
        }
    }
    
    // 思考动画
    function animateThinkingDots() {
        const dots = document.querySelectorAll('.thinking-dots span');
        let i = 0;
        dots.forEach((dot, index) => {
            setTimeout(() => {
                dot.style.opacity = 1;
                setTimeout(() => {
                    dot.style.opacity = 0.3;
                }, 300);
            }, index * 300);
        });
    }

    // 页面初始化
    function init() {
        // 添加样式到思考动画
        const style = document.createElement('style');
        style.textContent = `
            .thinking-dots span {
                opacity: 0.3;
                font-size: 1.5rem;
                animation: thinking 1.5s infinite;
                display: inline-block;
                margin-right: 2px;
            }
            .thinking-dots span:nth-child(2) {
                animation-delay: 0.5s;
            }
            .thinking-dots span:nth-child(3) {
                animation-delay: 1s;
            }
            @keyframes thinking {
                0% { opacity: 0.3; }
                50% { opacity: 1; }
                100% { opacity: 0.3; }
            }
        `;
        document.head.appendChild(style);
    }
    
    init();
}); 