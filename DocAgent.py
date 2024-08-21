import os
from dotenv import load_dotenv
from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import OllamaEmbeddings
from langchain.chains import create_retrieval_chain
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


# 接claude
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
    根據供應商名稱查詢建檔規則
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
    print(output)

    return output


@tool
def check_weather(location: str, at_time: datetime | None = None) -> float:
    """Return the weather forecast for the specified location."""
    return f"It's always sunny in {location}"


tools = [check_weather, check_rule_by_supplier_name, compare_files_with_llm]

# Prompt
# prompt = hub.pull('hwchase17/openai-functions-agent')
# prompt = hub.pull("rlm/rag-prompt")
RAG_system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

RAG_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_system_prompt),
        ("human", "{input}"),
    ]
)


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

# RAG chain
# question_answer_chain = create_stuff_documents_chain(gpt35Model, RAG_prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)
# response = rag_chain.invoke({"input": "這本書的書名是什麼？"})
# print(response["answer"])
# for document in response["context"]:
#     print(document)
#     print()
# for chunk in rag_chain.stream("這本書的書名是什麼？"):
#     print(chunk, end="", flush=True)
# retriever_tool = create_retriever_tool(
#     retriever,
#     "langsmith_search",  # name
#     "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",  # description
# )
# chain = prompt | claudeModel | StrOutputParser()
# with_message_history = RunnableWithMessageHistory(
#     chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
#     # output_messages_key="answer",
# )

system_prompt = "You are an assistant for retrieve data."
graph = create_react_agent(gpt35Model, tools, state_modifier=system_prompt)
inputs = {
    "messages": [("user", "這本書的書名是什麼？")]
    + [("human", "file_path: ./doc/上誼.docx")]
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
