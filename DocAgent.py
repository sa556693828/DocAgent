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
from datetime import datetime

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")


# LLM
claudeModel = ChatAnthropic(model_name="claude-3-sonnet-20240229")
gpt35Model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# ollamaModel = Ollama(model="llama2")

# Memory
memory = MemorySaver()
config = {"configurable": {"session_id": "abc123"}}


store = {}


def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")


# tools
@tool
def check_rule_by_supplier_name(supplier_name):
    """
    根據出版商名稱查詢建檔規則
    """
    client = MongoClient(
        "mongodb+srv://tommy:pN3hJwrbAnb4ESoV@test1.fjnut.mongodb.net/"
    )
    db = client["test"]
    collection = db["rules"]
    query = {"supplier_name": supplier_name}
    results = collection.find(query)
    rules_list = []
    for result in results:
        rules_list.append(result.get("rule", []))
    return rules_list


@tool
def compare_files_with_llm(file_path1):
    """
    比較用戶提供的文件與標準格式，生成轉換規則
    """
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
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
    並且請你學習建檔規則：{ruleData}，學習每個欄位的建檔規則。
    用戶會給你一個出版社的書籍資訊，請你按照標準欄位、建檔規則，學習該出版社資料的欄位轉換成標準欄位的轉換規則給我。
    """

    compare_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                compare_prompt,
            ),
            (
                "human",
                "請產生我這個出版商資料轉換成標準欄位的轉換規則{file}，回傳規則就好，要讓相同出版社的資料，透過此規則都能轉換成標準欄位",
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
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    userFile = Docx2txtLoader(file_path1)
    userFileData = userFile.load()
    data = text_splitter.split_documents(userFileData)
    system_prompt = "You are an assistant for retrieve data!"

    retrieve_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            (
                "human",
                "幫我根據這份檔案{file}找出出版商名稱",
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
]

# Prompt
# prompt = hub.pull('hwchase17/openai-functions-agent')
# prompt = hub.pull("rlm/rag-prompt")

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             # "Based on the table schema below, write a SQL query that would answer the user's question: {db_schema}",
#             "You are a helpful assistant.",
#         ),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("user", "{input}"),
#     ]
# )

# Use the agent
# chain = prompt | claudeModel | StrOutputParser()
# with_message_history = RunnableWithMessageHistory(
#     chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
#     # output_messages_key="answer",
# )

system_prompt = """
    根據用戶資料中的出版商名稱查詢建檔規則，
    如果有查到就只回傳建檔規則不要加任何文字，
    如果沒有就比較用戶提供的文件與標準格式生成轉換規則，
    並且回傳轉換規則
"""
graph = create_react_agent(gpt35Model, tools, state_modifier=system_prompt)
inputs = {
    "messages": [("user", "此為新書資料")] + [("human", "file_path: ./doc/上誼.docx")]
}

for s in graph.stream(inputs, stream_mode="values"):
    message = s["messages"][-1]
    if isinstance(message, tuple):
        print(message)
    else:
        message.pretty_print()

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
