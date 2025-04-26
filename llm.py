import time
import os
import requests
import json
from basereal import BaseReal
from logger import logger

def llm_response(message, nerfreal: BaseReal, model_name="llama3", ollama_url="http://localhost:11434"):
    """
    使用本地Ollama模型处理用户消息并将结果发送到数字人
    
    参数:
        message: 用户输入的消息
        nerfreal: 数字人对象
        model_name: Ollama中的模型名称，默认"llama3"
        ollama_url: Ollama服务地址，默认"http://localhost:11434"
    """
    start = time.perf_counter()
    logger.info(f"使用Ollama模型: {model_name}")
    
    # Ollama API地址
    api_url = f"{ollama_url}/api/chat"
    
    # Ollama请求数据
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "你是一个有帮助的中文助手，请用简短、直接的方式回答问题。"},
            {"role": "user", "content": message}
        ],
        "stream": True
    }
    
    end = time.perf_counter()
    logger.info(f"llm Time init: {end-start}s")
    
    try:
        # 发送流式请求
        response = requests.post(api_url, json=data, stream=True)
        
        if response.status_code != 200:
            error_msg = f"Ollama API错误: {response.status_code} - {response.text}"
            logger.error(error_msg)
            nerfreal.put_msg_txt(error_msg)
            return error_msg
        
        result = ""
        first = True
        
        # 处理流式响应
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    if 'message' in chunk and 'content' in chunk['message']:
                        msg = chunk['message']['content']
                        
                        if first:
                            end = time.perf_counter()
                            logger.info(f"llm Time to first chunk: {end-start}s")
                            first = False
                        
                        lastpos = 0
                        for i, char in enumerate(msg):
                            if char in ",.!;:，。！？：；":
                                result = result + msg[lastpos:i+1]
                                lastpos = i+1
                                if len(result) > 10:
                                    logger.info(result)
                                    nerfreal.put_msg_txt(result)
                                    result = ""
                        result = result + msg[lastpos:]
                except json.JSONDecodeError:
                    logger.error(f"解析JSON失败: {line}")
                    continue
                
        # 发送剩余部分
        if result:
            nerfreal.put_msg_txt(result)
            
        end = time.perf_counter()
        logger.info(f"llm Time to last chunk: {end-start}s")
        
        return "处理完成"
    
    except Exception as e:
        error_msg = f"调用Ollama出错: {str(e)}"
        logger.error(error_msg)
        nerfreal.put_msg_txt(error_msg)
        return error_msg    