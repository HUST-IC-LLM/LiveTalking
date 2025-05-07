'''
接入RAGFlow，封装HTTP接口，
固定模型信息，
进行Assitant和Session、Databases的管理
'''

from http.client import HTTPException
import json
from pathlib import Path
from re import L
from typing import Optional
import httpx

from .schem import ChatAssistantConfig, Response_Chat, Response_GetSessions

from config import Config

class RAGFlowClient:
    '''封装RAGFlow API， 将HTTP API接口参数进行封装'''
    def __init__(self, base_url: str=Config.RAGFLOW_BASE_URL, api_key: str = Config.RAGFLOW_API_KEY):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    async def getAssistantList(self, filter_params:dict = {}, is_agent : bool = False)-> dict:
        '''获取助理列表'''
        if is_agent:
            url = f"{self.base_url}/api/v1/agents"
        else:
            url = f"{self.base_url}/api/v1/chats"
            
        if filter_params.get("page"):
            url += f'?page={filter_params["page"]}"'
        else:
            url += f"?page=1"
        if filter_params.get("page_size"):
            url += f'&page_size={filter_params["page_size"]}'
        if filter_params.get("orderby"):
            url += f'&orderby={filter_params["orderby"]}'
        if filter_params.get("desc"):
            url += f'&desc={filter_params["desc"]}'
        if filter_params.get("chat_name"):
            url += f'&name={filter_params["chat_name"]}'
        if filter_params.get("chat_id"):
            url += f'&id={filter_params["chat_id"]}'
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(
                url,
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()
    async def createAssistant(self, assistant : ChatAssistantConfig)-> dict:
        "根据指定的知识库创建助理"
        payload = assistant.__dict__
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/chats",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
    async def updateAssistant(self, assitant_id : str, updated_params : dict)->dict:
        '''
        修改assitant_id指定的助理配置
        @param args : 需要修改的参数字典
        '''
        payload = updated_params
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.put(
                f"{self.base_url}/api/v1/chats/{assitant_id}",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
    async def addDatasetsToAssistant(self, assistant_id:str, dataset_ids:list[str])->dict:
        '''将特定的datasets加到对应assistant_id中'''
        response = await self.getAssistantList({"chat_id":assistant_id})                  # 先查询对应的assitant
        raw_dataset_ids : list[str] = response["data"]["dataset_ids"]               # 提取原始数据库组
        response = await self.updateAssistant(assistant_id,raw_dataset_ids + dataset_ids) # 添加数据库组到assistant
        return response
        
    async def deleteAssistant(self, assistant_ids: list[str])->dict:
        "根据助理id列表批量删除对应助理"
        payload = {"ids":assistant_ids}
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.delete(
                f"{self.base_url}/api/v1/chats",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
    async def createSession(self,assistant_id : str, name : str = "test", user_id : str | None = None, is_agent : bool = False)-> dict:
        "在指定助理基础上开启会话"
        if is_agent:
            url = f"{self.base_url}/api/v1/agents/{assistant_id}/sessions"
        else:
            url = f"{self.base_url}/api/v1/chats/{assistant_id}/sessions"
        payload = {
            "name" : name,
            "user_id" : user_id
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
    async def chat(self, assistant_id: str, question: str, session_id : Optional[str] = None, user_id : Optional[str] = None, stream: bool = False, is_agent: bool = False):
        """对指定assistant，在指定会话中（若空则新建后再）进行一次对话"""
        if is_agent:
            url = f"{self.base_url}/api/v1/agents/{assistant_id}/completions"
        else:
            url = f"{self.base_url}/api/v1/chats/{assistant_id}/completions"
        payload = {
            "question": question,
            "stream": stream, 
            "session_id" : session_id,
            "user_id" : user_id
        }
        
        if stream:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST",
                                    url,
                                    headers=self.headers,
                                    json=payload) as response:
                    response.raise_for_status()
                    session_id = ""
                    async for line in response.aiter_lines():
                        if line:
                            line = line.strip()
                            if line.startswith('data:'):
                                data = json.loads(line[5:].strip())
                                if data.get('code') == 0:
                                    if isinstance(data.get('data'), bool):
                                        # 流式结束标记
                                        yield json.dumps({
                                            'type': 'end',
                                            'session_id' : session_id
                                            })
                                    else:
                                        # 正常消息
                                        session_id = data["data"]["session_id"]
                                        yield json.dumps({
                                            'type': 'text',
                                            'content': data['data']['answer'],
                                            'session_id' : session_id
                                        })
        else:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    url,
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                print("[Debug] Chat Response:",response.json())
                response = response.json()
                def parse_sse_data(sse_str: str) -> dict:
                    # 去除前缀和换行符
                    json_str = sse_str.strip().replace("data:", "", 1)
                    return json.loads(json_str)
                if response['data'] and type(response['data']) == str: # 说明新创建了一个会话
                    parsed_data = parse_sse_data(response["data"])
                    parsed_response = Response_Chat(**{
                        **response,
                        "data": parsed_data["data"]  # 提取嵌套的 data 字段
                    })
                    print("[Debug] parsed_response:",parsed_response)
                    if question == "":
                        yield parsed_response
                    # 如果question不为空，重新发送question到现在会话
                    payload["session_id"] = parsed_response.data.session_id
                    response = await client.post(
                        url,
                        headers=self.headers,
                        json=payload
                    )
                    yield Response_Chat(**response.json())
                if question == "": # can't send empty message to existing session
                    yield HTTPException()
                yield Response_Chat(**response.json())

    # def upload_document(self, kb_id: str, file_path: str) -> str:
    #     """上传文档"""
    #     files = {'file': open(file_path, 'rb')}
    #     data = {'knowledge_base_id': kb_id}
    #     response = requests.post(
    #         f"{self.base_url}/api/v1/upload",
    #         headers={"Authorization": self.headers["Authorization"]},
    #         files=files,
    #         data=data
    #     )
    #     response.raise_for_status()
    #     return response.json()["task_id"]  # 返回异步任务ID

    async def getSessionList(self, assistant_id : str, 
                       page : int = 1,
                       page_size : int = 30,
                       user_id : str = "",
                       session_id : str = "",
                       session_name : str = "",
                       orderby : str = "create_time",
                       desc : bool = False
                       )-> Response_GetSessions:
        '''获取指定assitant的会话列表（按页访问），并提供筛选条件（会话id/会话名、排序方式等信息）'''
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(
                f"{self.base_url}/api/v1/chats/{assistant_id}/sessions?page={page}&page_size={page_size}&orderby={orderby}&desc={desc}&name={session_name}&id={session_id}&user_id={user_id}",
                headers=self.headers
            )
            response.raise_for_status()
            
            print("[Debug] getChatSession done:",str(response.json())[:200])
            return Response_GetSessions(**response.json())
    
    async def deleteSession(self,assitant_id:str,
                      ids : list[str] = [],
                      is_agent : bool = False
                      )-> dict:
        "删除assitant_id指定的assitant下，ids列表对应的所有会话，如果为空，则删除所有会话"
        if is_agent:
            url = f"{self.base_url}/api/v1/agents/{assitant_id}/sessions"
        else:
            url = f"{self.base_url}/api/v1/chats/{assitant_id}/sessions"
        payload = {"ids" : ids}
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.delete(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
    
    async def uploadDocuments(self,dataset_id:str,file_path:str):
        """
        上传文件到 RAGFlow 指定数据集
        
        :param dataset_id: 目标数据集 ID
        :param file_path: 本地文件路径
        :param api_key: RAGFlow API 密钥
        :return: 响应结果
        """
        print("[Debug] uploadDocuments...")
        async with httpx.AsyncClient(timeout=60.0) as client:
            with open(file_path, "rb") as f:
                files = [("file",(Path(file_path).name , f))]  # 保留原始文件名
                response = await client.post(
                    f"{self.base_url}/api/v1/datasets/{dataset_id}/documents",
                    headers={"Authorization": self.headers["Authorization"]},
                    files=files
                )
                print("[Debug] upload done",response.json())
                response.raise_for_status()
                return response.json()
    
    async def createDataset(self,name:str,description:str=""):
        '''
        创建知识库
        '''
        print("[Debug] createDataset...")
        payload = {
            "name" : name,
            "description" : description
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/datasets",
                headers=self.headers,
                json = payload
            )
            print("[Debug] createDataset Done:",response.json())
            return response.json()
    async def parseDocuments(self,dataset_id,document_ids:list[str]):
        print("[Debug] parseDocuments...")
        payload = {
            "document_ids": document_ids
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/datasets/{dataset_id}/chunks",
                headers=self.headers,
                json = payload
            )
            response.raise_for_status
            print("[Debug] parseDocuments done:",response)
            return response.json()
        
        

        

# 初始化客户端
rag_client = RAGFlowClient()

base_knowledge_base = "1658cfa410ac11f08f100242ac130006" # 基本知识库ids

# ================================== 接口函数 ======================================================================

# def handle_user_question(question: str, session_id : Optional[str] = None, kb_id : str = base_knowledge_base) -> str:
#     assitant : ChatAssistantConfig = ChatAssistantConfig(name = "test",dataset_ids=[kb_id])
#     try:
#         response = rag_client.getAssistantList()
#         print("[DEBUG] RAGFlow 助理列表:", response)  # 新增此行
#         assitant_id = ""
#         for d in response["data"]:
#             if "name" in d and d["name"]==assitant.name:
#                 assitant_id = d["id"]
#                 print("[DEBUG] RAGFLOW Already Exited Assitant:",d)
#         if not assitant_id:
#             response = rag_client.createAssistant(assitant)
#             assitant_id = response["data"]["id"]
#             print("[DEBUG] RAGFLOW 创建助理:",response)
            
#         response = rag_client.chat(assitant_id,question,session_id)
#         print("[DEBUG] RAGFLOW 对话:",response)
#         return response["data"]["answer"]
#     except requests.exceptions.HTTPError as e:
#         print("[ERROR] HTTP 请求失败:", e.response.text)  # 输出详细错误
#         return f"请求失败：{e.response.text}"
    
    