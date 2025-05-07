#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : ${DATE} ${TIME}
# @Author : jqjiksina
# @File : ${NAME}.py
'''
和RAGFlow的HTTP-API接口规范
'''

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Literal, Optional, Union
    
class VariableItem(BaseModel):
    """Variables to use in the 'System' field"""
    key: str = Field(..., description="Variable key name")
    optional: bool = Field(True, description="Whether the variable is optional")

class PromptConfig(BaseModel):
    """LLM prompt instructions configuration"""
    similarity_threshold: float = Field(default=0.2, ge=0, le=1, 
        description="Threshold for similarity between query and chunks")
    
    keywords_similarity_weight: float = Field(default=0.7, ge=0, le=1,
        description="Weight of keyword similarity in hybrid score")
    
    top_n: int = Field(default=6, ge=1,
        description="Number of top chunks fed to LLM")
    
    variables: List[VariableItem] = Field(
        default_factory=lambda: [VariableItem(key="knowledge", optional=True)],
        description="Variables for system prompt (curly bracketed)"
    )
    
    rerank_model: Optional[str] = Field(default=None,
        description="Reranking model name (default: use vector cosine)")
    
    top_k: int = Field(default=1024, ge=1,
        description="Top-k items for reranking")
    
    empty_response: str = Field(default="",
        description="Fallback response when no chunks retrieved")
    
    opener: str = Field(default="Hi! I am your assistant, can I help you?",
        description="Initial greeting message")
    
    show_quote: bool = Field(default=True,
        description="Display text source attribution")
    
    prompt: Optional[str] = Field(default=None,
        description="Actual prompt content for LLM")

class LLMConfig(BaseModel):
    """Large Language Model configuration"""
    model_name: Optional[str] = Field(default=None,
        description="Chat model name (default: user's default model)")
    
    temperature: float = Field(default=0.1, ge=0, le=2,
        description="Controls prediction randomness (0=deterministic)")
    
    top_p: float = Field(default=0.3, ge=0, le=1,
        description="Nucleus sampling threshold")
    
    presence_penalty: float = Field(default=0.4, ge=-2, le=2,
        description="Penalty for repeating existing content")
    
    frequency_penalty: float = Field(default=0.7, ge=-2, le=2,
        description="Penalty for frequent word repetition")
    
class ChatAssistantConfig(BaseModel):
    """Complete chat assistant configuration"""
    name : str = Field(default=...,
        description="The name of the chat assistant")
    avatar : Optional[str] = Field(default=None,
        description="Base64 encoding of the avatar")
    dataset_ids : Optional[List[str]] = Field(default=None,
        description="The IDs of the associated datasets")
    llm: LLMConfig = Field(default_factory=lambda:LLMConfig(),
        description="LLM settings with auto-generated defaults")
    prompt: PromptConfig = Field(default_factory=lambda:PromptConfig(),
        description="Prompt instructions with auto-generated defaults")
    
    

# ============================================= Datasets ============================================
# ========== 枚举定义 ==========
class ChunkMethod(str, Enum):
    NAIVE = "naive"
    MANUAL = "manual"
    QA = "qa"
    TABLE = "table"
    PAPER = "paper"
    BOOK = "book"
    LAWS = "laws"
    PRESENTATION = "presentation"
    PICTURE = "picture"
    ONE = "one"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    EMAIL = "email"
    
class Permission(str, Enum):
    ME = "me"
    TEAM = "team"

# ========== Datasets 嵌套配置模型 ==========
class RaptorConfig(BaseModel):
    use_raptor: bool = Field(default=False)

# ----- Datasets 通用配置基类 -----
class BaseParserConfig(BaseModel):
    pass

# ----- Datasets 不同 chunk_method 的配置模型 -----
class NaiveParserConfig(BaseParserConfig):
    chunk_token_count: int = Field(default=128, ge=1)
    layout_recognize: bool = Field(default=True)
    html4excel: bool = Field(default=False)
    delimiter: str = Field(default=r"\n!?。；！？")
    task_page_size: int = Field(default=12, ge=1)
    raptor: RaptorConfig = Field(default_factory=RaptorConfig)

class QAParserConfig(BaseParserConfig):
    raptor: RaptorConfig = Field(default_factory=RaptorConfig)

class KnowledgeGraphParserConfig(BaseParserConfig):
    chunk_token_count: int = Field(default=128, ge=1)
    delimiter: str = Field(default=r"\n!?。；！？")
    entity_types: List[str] = Field(
        default=["organization", "person", "location", "event", "time"],
        examples=["organization", "person"]
    )

class EmptyParserConfig(BaseParserConfig):
    pass

# ========== Datasets 主模型 ==========
class DatasetCreate(BaseModel):
    chunk_method: ChunkMethod = Field(
        default=ChunkMethod.NAIVE,
        description="Document chunking method strategy"
    )
    
    parser_config: Union[
        NaiveParserConfig,
        QAParserConfig,
        KnowledgeGraphParserConfig,
        EmptyParserConfig
    ] = Field(
        default_factory=NaiveParserConfig,
        description="Parser configuration depends on chunk_method"
    )

    # 动态验证配置匹配
    @validator('parser_config', pre=True, always=True)
    def validate_parser_config(cls, v, values):
        chunk_method = values.get('chunk_method', ChunkMethod.NAIVE)
        
        config_map = {
            ChunkMethod.NAIVE: NaiveParserConfig,
            ChunkMethod.MANUAL: QAParserConfig,
            ChunkMethod.QA: QAParserConfig,
            ChunkMethod.PAPER: QAParserConfig,
            ChunkMethod.BOOK: QAParserConfig,
            ChunkMethod.LAWS: QAParserConfig,
            ChunkMethod.PRESENTATION: QAParserConfig,
            ChunkMethod.TABLE: EmptyParserConfig,
            ChunkMethod.PICTURE: EmptyParserConfig,
            ChunkMethod.ONE: EmptyParserConfig,
            ChunkMethod.KNOWLEDGE_GRAPH: KnowledgeGraphParserConfig,
            ChunkMethod.EMAIL: EmptyParserConfig
        }
        
        config_cls = config_map[chunk_method]
        
        if isinstance(v, dict):
            return config_cls(**v)
        elif isinstance(v, config_cls):
            return v
        else:
            raise ValueError(
                f"Invalid parser_config for {chunk_method}, "
                f"expected {config_cls.__name__}"
            )
            
# ======================= Response ===============================
# ====== Response GetSessions ======
class Message(BaseModel):
    """单条聊天消息"""
    role: Literal["user", "assistant", "system"] = Field(
        default=...,
        description="消息角色 (user/assistant/system)"
    )
    content: str = Field(..., description="消息内容")

class SessionDataItem(BaseModel):
    """单个会话数据项"""
    chat_id: str = Field(default=..., description="会话唯一标识符")
    create_date: str = Field(default=..., description="创建日期 (RFC 2822 格式)")
    create_time: int = Field(default=..., description="创建时间戳 (毫秒)")
    id: str = Field(default=..., description="数据项唯一ID")
    messages: List[Message] = Field(default=..., description="消息列表")
    name: str = Field(default=..., description="会话名称")
    update_date: str = Field(default=..., description="最后更新日期")
    update_time: int = Field(default=..., description="最后更新时间戳 (毫秒)")

class Response_GetSessions(BaseModel):
    """GetSession API响应结构"""
    code: Literal[0, 1] = Field(..., description="状态码 (0=成功, 1=失败)")
    data: List[SessionDataItem] = Field(..., description="数据负载")
    
# ====== Response Chat ======
class Chunk(BaseModel):
    id: str
    content: str
    document_id: str = Field(..., alias="document_id")
    document_name: str
    dataset_id: str
    image_id: str = ""
    similarity: float
    vector_similarity: float = 0.0
    term_similarity: float = 1.0
    positions: List[str] = [""]

class DocAgg(BaseModel):
    doc_name: str
    doc_id: str
    count: int

class Reference(BaseModel):
    total: int
    chunks: List[Chunk]
    doc_aggs: List[DocAgg] = Field(..., alias="doc_aggs")

class SuccessData(BaseModel):
    answer: str
    reference: Union[Reference, Dict] = Field(default_factory=dict)
    audio_binary: Optional[bytes] = Field(default=None)
    id: Optional[str] = Field(default=None)
    session_id: str = Field(default=...)
    prompt: Optional[str] = Field(default=None)

class Error_Chat(BaseModel):
    code: int = Field(..., examples=[102])
    message: str = Field(..., examples=["Please input your question."])

class Response_Chat(BaseModel):
    code: int
    message: str = ""
    data: Optional[Union[SuccessData, bool]] = None

    @classmethod
    def success(cls, response : "Response_Chat")-> bool:
        return response.code == 0 and (type(response.data)==SuccessData)
    
    @classmethod
    def is_stream_end(cls, response : "Response_Chat")-> bool:
        return response.code == 0 and response.data == True
    
    @classmethod
    def is_error(cls, response : "Response_Chat")-> bool:
        return response.code == 102 and response.data == None
            


# ========== datasets 使用示例 ==========
# 默认配置（naive）
# naive_config = DatasetCreate()
# print("Naive Default:\n", naive_config.model_dump_json(indent=2))

# # 知识图谱配置
# kg_config = DatasetCreate(
#     chunk_method=ChunkMethod.KNOWLEDGE_GRAPH,
#     parser_config=KnowledgeGraphParserConfig(entity_types=["person", "location"])
# )
# print("\nKnowledge Graph:\n", kg_config.model_dump_json(indent=2))

# # 表格配置（空对象）
# table_config = DatasetCreate(
#     chunk_method=ChunkMethod.TABLE,
#     parser_config=EmptyParserConfig()
# )
# print("\nTable Config:\n", table_config.model_dump_json(indent=2))

# ========== ChatAssistantConfig 使用示例 ==========
# config = ChatAssistantConfig()
# print(config.model_dump())