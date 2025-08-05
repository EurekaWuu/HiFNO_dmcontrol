import socket
import json
import base64
import io
import time
import logging
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
import torch
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("llava_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LLaVA-Server")

# 禁用PyTorch初始化
disable_torch_init()

# 模型路径
model_path = "/mnt/lustre/GPU4/home/wuhanpeng/models/llava-v1.5-7b"
logger.info(f"加载模型: {model_path}")

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path=model_path, model_base=None, model_name=model_name
)
logger.info("模型加载完成")

# 功能类型与提示词映射
FEATURE_PROMPTS = {
    "general": "你是杏林千问健康助手，请以专业医疗人员的身份回答下面的问题，注重科学性和准确性，语言温和，易于理解。",
    "tcm": "你是杏林千问中医药专家，请从中医药理论和实践的角度回答下面的问题，可以引用经典医籍和现代研究，注重传统与现代结合。",
    "weight": "你是杏林千问体重管理专家，针对下面的问题，从饮食、运动、心理等多角度提供科学的体重管理建议，关注健康而非单纯减肥。",
    "chronic": "你是杏林千问慢病管理专家，针对下面的问题，提供慢性病预防、监测和管理的专业建议，强调生活方式干预和规范用药。",
    "guide": "你是杏林千问导诊专家，根据下面描述的症状，分析可能的疾病，并给出就医建议，包括应该挂哪个科室，需要做什么检查等。"
}

# 图像分析提示词映射
IMAGE_FEATURE_PROMPTS = {
    "general": "请分析这张医疗相关图片，提供专业的健康解读和建议。",
    "tcm": "请从中医角度分析这张图片，可以是舌诊、面诊或其他中医诊断图像，给出中医理论解释和建议。",
    "weight": "请分析这张与体重管理相关的图片，可能是体型、食物或运动图像，给出针对性的体重管理建议。",
    "chronic": "请分析这张与慢性病相关的图片，可能是检查报告、症状表现或生活方式，给出慢病管理建议。",
    "guide": "请分析这张医疗症状或检查结果图片，帮助判断可能的疾病，并给出就医建议。"
}

# 处理一个客户请求的函数
def process_request(data_json):
    start_time = time.time()
    try:
        # 解析请求数据
        request = json.loads(data_json)
        logger.info(f"收到请求: prompt长度={len(request.get('prompt', ''))}, 是否包含图像={('image_base64' in request)}")
        
        # 获取提示和功能类型
        prompt = request.get("prompt", "")
        feature = request.get("feature", "general")
        
        # 根据功能类型选择系统提示
        system_prompt = FEATURE_PROMPTS.get(feature, FEATURE_PROMPTS["general"])
        
        # 初始化图像
        image = None
        
        # 如果有图像，处理图像
        if "image_base64" in request:
            try:
                img_base64 = request["image_base64"]
                image = Image.open(io.BytesIO(base64.b64decode(img_base64))).convert("RGB")
                logger.info(f"图像解码成功，尺寸: {image.size}")
                
                # 为图像分析增加特定提示词
                image_prompt = IMAGE_FEATURE_PROMPTS.get(feature, IMAGE_FEATURE_PROMPTS["general"])
                if not prompt or prompt == "请分析这张图片":
                    prompt = image_prompt
                else:
                    prompt = f"{image_prompt} 具体问题: {prompt}"
            except Exception as e:
                logger.error(f"图片解码失败: {e}")
                return json.dumps({"error": f"图片解码失败: {str(e)}"})
        
        # 如果没有提示词和图像，返回错误
        if not prompt and not image:
            logger.error("请求中没有提供提示词或图像")
            return json.dumps({"error": "请求中需要提供提示词或图像"})
        
        # 开始模型推理
        logger.info("开始模型推理...")
        
        # 处理图像（如果有）
        image_tensor = None
        if image:
            image_tensor = process_images([image], image_processor, model.config)
            model_dtype = next(model.parameters()).dtype
            image_tensor = image_tensor.to(model_dtype).cuda()
        
        # 构建对话
        conv = conv_templates["llava_v1"].copy()
        
        # 添加系统提示
        if system_prompt:
            conv.system = system_prompt
        
        # 添加用户消息
        if image:
            conv.append_message(conv.roles[0], "<image>\n" + prompt)
        else:
            conv.append_message(conv.roles[0], prompt)
        
        # 添加助手空回复
        conv.append_message(conv.roles[1], None)
        
        # 编码输入
        prompt_text = conv.get_prompt()
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").cuda()
        
        # 记录输入
        logger.debug(f"模型输入: prompt='{prompt_text}', input_ids.shape={input_ids.shape}")
        if image:
            logger.debug(f"图像尺寸: {image.size}, 张量形状: {image_tensor.shape}")
        
        # 推理
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=1024,  # 增加最大输出长度
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        output = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        logger.info(f"生成回复，长度: {len(output)}")
        
        # 计算耗时
        duration = time.time() - start_time
        logger.info(f"请求处理完成，耗时: {duration:.2f}秒")
        
        # 返回结果
        response = {
            "response": output,
            "duration": round(duration, 2)
        }
        return json.dumps(response)
    
    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}", exc_info=True)
        return json.dumps({"error": f"处理请求时发生错误: {str(e)}"})

# 主服务器循环
def run_server():
    HOST, PORT = 'localhost', 7788
    
    logger.info(f"启动LLaVA Socket服务器，监听地址: {HOST}:{PORT}")
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # 设置socket选项
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(5)  # 允许最多5个排队连接
        
        logger.info(f"[LLaVA Socket服务器] 正在监听 {HOST}:{PORT}")
        
        while True:
            try:
                conn, addr = s.accept()
                logger.info(f"收到连接，来自: {addr}")
                
                with conn:
                    # 接收数据
                    data = b''
                    while True:
                        packet = conn.recv(4096)
                        if not packet:
                            break
                        data += packet
                    
                    if len(data) == 0:
                        logger.warning("接收到空数据")
                        continue
                    
                    logger.info(f"接收数据完成，大小: {len(data)} 字节")
                    
                    # 处理请求
                    response_json = process_request(data)
                    
                    # 发送响应
                    conn.sendall(response_json.encode())
                    logger.info("响应已发送")
            
            except Exception as e:
                logger.error(f"处理连接时发生错误: {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("杏林千问 LLaVA 服务器启动")
    run_server()


"""
CUDA_VISIBLE_DEVICES=0 python llava_socket_server.py

"""