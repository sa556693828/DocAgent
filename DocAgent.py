import os
from dotenv import load_dotenv
from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import Docx2txtLoader, CSVLoader
from pymongo import MongoClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import json

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
mongoUrl = os.environ["MONGO_URI"] = os.getenv("MONGO_URI")


# LLM
claudeModel = ChatAnthropic(model_name="claude-3-sonnet-20240229")
gpt4oModel = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# ollamaModel = Ollama(model="llama2")

# Memory
memory = MemorySaver()
config = {"configurable": {"session_id": "abc123"}}


store = {}


def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")


# tools
@tool
def check_rule_by_supplier_name(supplier_name) -> list:
    """
    根據出版商名稱查詢建檔規則
    """
    client = MongoClient(mongoUrl)
    db = client["DocAgent"]
    collection = db["suppliers"]
    query = {"supplier_name": supplier_name}
    results = collection.find(query)
    rules_list = []
    for result in results:
        rules_list.append(result.get("rule", []))
    return rules_list


@tool
def insert_rule_by_supplier_name(supplier_name: str, rule: str):
    """
    根據出版商名稱插入建檔規則
    """
    client = MongoClient(mongoUrl)
    db = client["DocAgent"]
    collection = db["suppliers"]
    new_data = {"supplier_name": supplier_name, "rule": rule}
    result = collection.insert_one(new_data)
    if result.inserted_id:
        return "success"
    else:
        return "fail"


@tool
def insert_collected_content(content: str):
    """Insert collected content into MongoDB"""
    client = MongoClient(mongoUrl)
    db = client["DocAgent"]
    collection = db["standard_form"]
    new_data = json.loads(content)
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
def retrieve_supplier_name(file_path1) -> str:
    """
    讀取用戶資料，找出出版商名稱
    """
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    userFile = Docx2txtLoader(file_path1)
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


tools = [
    retrieve_supplier_name,
    check_rule_by_supplier_name,
    compare_files_with_llm,
    insert_collected_content,
    insert_rule_by_supplier_name,
    transform_data,
]

# Prompt
# prompt = hub.pull('hwchase17/openai-functions-agent')
# prompt = hub.pull("rlm/rag-prompt")

system_prompt = """
    你是一個書商的建檔人員，請按照下面步驟做事：
    1.根據用戶提供的file，找出出版商名稱
    2.使用出版社名稱查詢建檔規則
    3-1.如果有查到 => 就使用此建檔規則，轉換用戶提供的文件
    3-2.如果沒有查到 => 就比較用戶提供的文件與標準格式生成轉換規則，並且將此規則insert進DB
    4.使用查到的或是生成好的建檔規則將用戶提供的文件轉換成標準格式
    5.將轉換後的資料insert進DB
    都成功就回傳 "success"
"""

# Use the agent
graph = create_react_agent(gpt4oModel, tools, state_modifier=system_prompt)
inputs = {
    "messages": [("user", "此為新書資料")]
    + [("human", "file_path: ./doc/九歌資料.docx")]
}

result = graph.invoke(inputs)

print(result)


# for s in graph.stream(inputs, stream_mode="values"):
#     message = s["messages"][-1]
#     if isinstance(message, tuple):
#         print(message)
#     else:
#         message.pretty_print()

# input_text = input(">>> ")
# while input_text.lower() != "bye":
#     if input_text:
#         response = with_message_history.invoke(
#             {
#                 "input": input_text,
#             },
#             config,
#         )
#         print(response)
#         # chat_history.append(HumanMessage(content=input_text))
#         # chat_history.append(AIMessage(content=response['answer']))
#     input_text = input(">>> ")
