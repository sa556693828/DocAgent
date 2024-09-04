import os
from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import Docx2txtLoader, CSVLoader, PyPDFLoader
from pymongo import MongoClient
import pymongo
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import json
from langserve import add_routes
from fastapi import Depends, FastAPI, Header, HTTPException
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from bson.objectid import ObjectId
from google.cloud import storage, exceptions as gcs_exceptions
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel
import certifi

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
mongoUrl = os.environ["MONGO_URI"] = os.getenv("MONGO_URI")
key = os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS"
)
# LLM
gpt4oModel = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# tools
@tool
def get_file_from_db_and_download(file_id: str):
    """
    根據file_id查詢文件，並從bucket中下載，回傳file路徑
    """
    try:
        client = MongoClient(mongoUrl, tlsCAFile=certifi.where())
        db = client["DocAgent"]
        collection = db["input_files"]
        query = {"_id": ObjectId(file_id)}
        results = collection.find_one(query)

        if not results or "name" not in results:
            return "錯誤：在數據庫中未找到指定的文件"

        file_name = results["name"]

        storage_client = storage.Client.from_service_account_json(key)
        bucket = storage_client.bucket("docagent_files")
        blob = bucket.blob(file_name)
        path = f"./newDoc/{file_name}"

        blob.download_to_filename(path)
        return path

    except pymongo.errors.PyMongoError as e:
        return f"數據庫錯誤：{str(e)}"
    except gcs_exceptions.GoogleCloudError as e:
        return f"Google Cloud Storage錯誤：{str(e)}"
    except Exception as e:
        return f"未知錯誤：{str(e)}"


@tool
def retrieve_supplier_name(file_path1) -> str:
    """
    讀取用戶資料，找出出版商名稱
    """
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)

    if file_path1.endswith(".docx"):
        userFile = Docx2txtLoader(file_path1)
    elif file_path1.endswith(".pdf"):
        userFile = PyPDFLoader(file_path1)
    else:
        raise ValueError("Unsupported file format")

    userFileData = userFile.load()
    data = text_splitter.split_documents(userFileData)
    system_prompt = """
    你是一個retriever助理，請根據用戶提供的file，找出出版商名稱
    """

    retrieve_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            (
                "human",
                "{file}",
            ),
        ]
    )
    retrieve_chain = retrieve_prompt_template | llm | StrOutputParser()
    output = retrieve_chain.invoke(
        {"file": data},
    )

    return output


@tool
def check_rule_by_supplier_name(supplier_name) -> list:
    """
    根據出版商名稱查詢建檔規則
    """
    client = MongoClient(mongoUrl, tlsCAFile=certifi.where())
    db = client["DocAgent"]
    collection = db["suppliers"]
    query = {"supplier_name": supplier_name}
    results = collection.find(query)
    rules_list = []
    for result in results:
        rules_list.append(result.get("rule", []))
    return rules_list


@tool
def compare_files_with_llm(file_path1) -> str:
    """
    比較用戶提供的文件與標準格式，生成轉換規則
    """
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    csv = CSVLoader(
        file_path="./doc/標準格式.csv",
        encoding="utf-8",
    )
    csvData = csv.load()
    # csv = text_splitter.split_documents(data)
    rules = Docx2txtLoader("./doc/建檔規則.docx")
    rulesData = rules.load()
    # doc = text_splitter.split_documents(rulesData)
    userFile = Docx2txtLoader(file_path1)
    userFileData = userFile.load()
    data = text_splitter.split_documents(userFileData)
    compare_prompt = """
    現在請你扮演一個書商的建檔人員，請你將此CSV檔案視為標準欄位：{csvData}
    並且請你學習建檔規則：{ruleData}，學習標準欄位中部分欄位的建檔規則。
    分析用戶給的檔案內的欄位、內容，學習該出版社資料的內容、欄位格式如何填入進標準欄位。
    例如：
    1.用戶格式為：書名而標準格式為：主要商品名稱，回傳 "書名=>主要商品名稱"
    2.用戶格式為：中文書名而標準格式為：主要商品名稱，回傳 "中文書名=>主要商品名稱"
    3.用戶格式為：尺寸(公分) 14*20*1.9 而標準格式為：商品長度 140、商品寬度 200、商品厚度 19，回傳 "尺寸(公分)=>商品長度、商品寬度、商品厚度 並將單位轉成mm"
    我需要所有欄位、內容、附件內容的對應規則，沒有內容的欄位也要。
    回傳格式：
    以下為...出版社的轉換規則
    1.欄位1=>標準欄位1
    2.欄位2=>標準欄位2
    3.欄位3=>標準欄位3
    ...
    """

    compare_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                compare_prompt,
            ),
            (
                "human",
                "{file}",
            ),
        ]
    )
    get_rule_chain = compare_prompt_template | llm | StrOutputParser()
    output = get_rule_chain.invoke(
        {"file": data, "csvData": csvData, "ruleData": rulesData},
        # config,
    )

    return output


@tool
def insert_rule_by_supplier_name(supplier_name: str, rule: str):
    """
    根據出版商名稱插入建檔規則
    """
    client = MongoClient(mongoUrl, tlsCAFile=certifi.where())
    db = client["DocAgent"]
    collection = db["suppliers"]
    new_data = {"supplier_name": supplier_name, "rule": rule}
    result = collection.insert_one(new_data)
    if result.inserted_id:
        return "success"
    else:
        return "fail"


@tool
def transform_data(file_path1: str, rule: str) -> str:
    """
    將用戶提供的文件內容轉換成標準格式並回傳JSON
    """
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    csv = CSVLoader(
        file_path="./doc/標準格式.csv",
        encoding="utf-8",
    )
    csvData = csv.load()
    # csv = text_splitter.split_documents(data)
    rules = Docx2txtLoader("./doc/建檔規則.docx")
    rulesData = rules.load()
    # doc = text_splitter.split_documents(rulesData)
    userFile = Docx2txtLoader(file_path1)
    userFileData = userFile.load()
    # data = text_splitter.split_documents(userFileData)
    # 現在請你扮演一個retriever，
    compare_prompt = """
    將此CSV檔案視為標準欄位：{csvData}
    並且學習通用建檔規則：{ruleData}
    以及針對此出版社的轉換規則：{rule}
    用戶提供的文件：{file}

    將用戶提供的文件內容轉換成標準格式，不要省略任何文字
    我要將完整的資料轉換成JSON格式，含有標準格式的所有欄位
    """
    compare_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                compare_prompt,
            ),
            (
                "human",
                "{file}",
            ),
        ]
    )
    get_rule_chain = compare_prompt_template | llm | StrOutputParser()
    output = get_rule_chain.invoke(
        {"file": userFileData, "csvData": csvData, "ruleData": rulesData, "rule": rule},
    )

    return output


@tool
def insert_collected_content(content: str):
    """Insert collected content into MongoDB"""
    client = MongoClient(mongoUrl, tlsCAFile=certifi.where())
    db = client["DocAgent"]
    collection = db["standard_form"]
    new_data = json.loads(content)
    result = collection.insert_one(new_data)
    if result.inserted_id:
        return "success"
    else:
        return "fail"


tools = [
    get_file_from_db_and_download,
    retrieve_supplier_name,
    check_rule_by_supplier_name,
    compare_files_with_llm,
    insert_rule_by_supplier_name,
    transform_data,
    insert_collected_content,
]

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                你是一個書商的建檔人員，請按照下面步驟做事：
                1.先去DB查詢用戶提供的file_id，如果存在就回傳此file_id的資料並將檔案下載下來，沒有就回覆用戶找不到
                2.根據用戶提供的file，找出出版商名稱
                3.使用出版社名稱查詢建檔規則
                4-1.如果有查到 => 就使用此建檔規則，轉換用戶提供的文件
                4-2.如果沒有查到 => 就比較用戶提供的文件與標準格式生成轉換規則，並且將此規則insert進DB
                5.使用查到的或是生成好的建檔規則將用戶提供的文件轉換成標準格式
                6.將轉換後的資料insert進DB
                都成功就回傳 "success"
            """,
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


# Use the agent
agent = create_tool_calling_agent(gpt4oModel, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

app = FastAPI(
    title="DocAgent",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


async def verify_api_key(authorization: Optional[str] = Header(None)):
    if authorization is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    # assuming the token is provided as a Bearer token
    api_key = (
        authorization.split(" ")[1] if len(authorization.split(" ")) == 2 else None
    )
    if api_key is None:
        raise HTTPException(
            status_code=401, detail="Invalid Authorization header format"
        )

    if api_key != "valid_api_key":
        raise HTTPException(status_code=403, detail="Invalid API Key")

    return {"user_name": "John"}


class Input(BaseModel):
    input: str


class Output(BaseModel):
    output: Any


add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output).with_config(
        {"run_name": "agent"}
    ),
    path="/docAgent",
    # dependencies=[Depends(verify_api_key)],
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=9000)
