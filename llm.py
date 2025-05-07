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
        "messages": [   # 如果要追加以前的消息记录，则需要加上以前的message进行管理！
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
                                if len(result) > 10:    # 以10个字符为单位，发送给数字人说话
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
    

def ragflow_response(message, nerfreal : BaseReal, ragflow_url = "http://localhost:8080", agent_id = "a4cf97b82a3311f0b9a9529bb6126436"):
    '''调用ragflow接口，'''
    from ragflow.ragflow import rag_client
    import asyncio
    import json
    
    logger.info(f"使用RAGFlow模型，Agent ID: {agent_id}")
    logger.info(f"RAGFlow URL: {ragflow_url}")
    
    try:
        # 创建异步事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # 获取异步生成器
        async_gen = rag_client.chat(assistant_id=agent_id, question=message, stream=True, is_agent=True)
        
        msg = "" # 流式响应收到的总消息
        lastpos = 0
        
        # 使用异步迭代器处理响应
        async def process_response():
            nonlocal msg, lastpos
            try:
                async for chunk in async_gen:
                    chunk_data = json.loads(chunk)
                    if chunk_data["type"] == "text":
                        msg = chunk_data["content"]
                        result = msg[lastpos:]
                        nerfreal.put_msg_txt(result) 
                        lastpos = len(msg)
                    elif chunk_data['type'] == "end":
                        logger.info("[Debug] ragflow_response end")
            except json.JSONDecodeError as e:
                error_detail = f"JSON解析错误: {str(e)}"
                logger.error(f"RAGFlow JSON解析错误: {error_detail}")
                raise Exception(error_detail)
            except Exception as e:
                error_detail = f"处理响应时出错: {str(e)}"
                logger.error(f"RAGFlow 处理错误: {error_detail}")
                raise Exception(error_detail)
        
        # 运行异步处理
        loop.run_until_complete(process_response())
        loop.close()
                
        return "ragflow 消息处理完成"
    
    except Exception as e:
        error_msg = f"调用RAGFlow出错: {str(e)}"
        logger.error(error_msg)
        nerfreal.put_msg_txt(error_msg)
        return error_msg
    
    