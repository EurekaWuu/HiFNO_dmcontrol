import socket
import json
import base64
import io
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
import torch


disable_torch_init()
model_path = "/mnt/lustre/GPU4/home/wuhanpeng/models/llava-v1.5-7b"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path=model_path, model_base=None, model_name=model_name
)


HOST, PORT = 'localhost', 7788
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"[LLaVA Socket Server] Listening on {HOST}:{PORT}")

    while True:
        conn, addr = s.accept()
        print(f"[Server] 收到连接来自: {addr}")
        with conn:
            data = b''
            while True:
                packet = conn.recv(4096)
                if not packet:
                    break
                data += packet
            print(f"[Server] 收到数据: {data[:100]}... 总长度: {len(data)} 字节")
            try:
                request = json.loads(data.decode())
            except Exception as e:
                print(f"[Server] JSON解析失败: {e}")
                response = {"error": f"JSON解析失败: {e}"}
                conn.sendall(json.dumps(response).encode())
                continue

            prompt = request.get("prompt", "Describe the image.")
            img_base64 = request["image_base64"]

            try:
                image = Image.open(io.BytesIO(base64.b64decode(img_base64))).convert("RGB")
                print(f"[Server][DEBUG] 原始图像尺寸: {image.size}")
            except Exception as e:
                print(f"[Server] 图片解码失败: {e}")
                response = {"error": f"图片解码失败: {e}"}
                conn.sendall(json.dumps(response).encode())
                continue

            print("[Server] 开始模型推理...")
            image_tensor = process_images([image], image_processor, model.config)
            model_dtype = next(model.parameters()).dtype
            image_tensor = image_tensor.to(model_dtype).cuda()

            conv = conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], "<image>\n" + prompt)
            conv.append_message(conv.roles[1], None)
            input_ids = tokenizer.encode(conv.get_prompt(), return_tensors="pt").cuda()

            print(f"[Server] prompt: {prompt}")
            try:
                prompt_text = conv.get_prompt()
                print("[Server][DEBUG] 构造的 prompt 文本为：", repr(prompt_text))
                input_ids = tokenizer.encode(prompt_text, return_tensors="pt").cuda()
                decoded_prompt = tokenizer.decode(input_ids[0].cpu().tolist())
                if decoded_prompt.strip() == "":
                    print("[Server][DEBUG] 解码结果为空")
                else:
                    print("[Server][DEBUG] 实际模型看到的prompt:", decoded_prompt)
            except Exception as e:
                print("[Server][DEBUG] decode 失败:", e)

                print(f"[Server][DEBUG] decode 失败: {e}")
            print(f"[Server] image size: {image.size}, mode: {image.mode}")
            print(f"[Server] image_tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
            print(f"[Server] input_ids shape: {input_ids.shape}")

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=512,
                    eos_token_id=tokenizer.eos_token_id
                )
            print("[Server] 推理完成，发送结果")
            output = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
            print(f"[Server] decode output: '{output}' (len={len(output)})")
            response = {"description": output}
            print(f"[Server] response: {response}")
            conn.sendall(json.dumps(response).encode())
            print("[Server] 结果已发送")



"""
CUDA_VISIBLE_DEVICES=0 python llava_socket_server.py



"""