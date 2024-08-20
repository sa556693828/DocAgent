import os
from dotenv import load_dotenv

# import requests
from langchain import hub
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.checkpoint import MemorySaver  # an in-memory checkpointer

# from langgraph.prebuilt import create_react_agent


load_dotenv()

# 从环境变量中获取API密钥
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

with_message_history = RunnableWithMessageHistory(model, get_session_history)


config = {"configurable": {"session_id": "abc3"}}


@tool
def check_site_alive(site: str) -> bool:
    """Check a site is alive or not."""
    try:
        # resp = requests.get(f"https://{site}")
        # resp.raise_for_status()
        print("check_site_alive")
        return True
    except Exception:
        return False


# Prompt Template
# prompt = hub.pull('hwchase17/openai-functions-agent')
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Based on the table schema below, write a SQL query that would answer the user's question: {db_schema}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "user",
            'Please generate a SQL query for the following question: "{input}". \
     The query should be formatted as follows without any additional explanation: \
     SQL> <sql_query>\
    ',
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
tools = [
    check_site_alive,
]


system_message = "You are a helpful assistant."

# 使用 LCEL 語法建立 chain
memory = MemorySaver()
app = create_react_agent(
    model, tools, state_modifier=system_message, checkpointer=memory
)

# new
# file_path = "./ResumeSomer.pdf"
# loader = file_path.endswith(".pdf") and PyPDFLoader(file_path) or TextLoader(file_path)
file_path = "./上誼.docx"
azuredoc_endpoint = "https://docagent.cognitiveservices.azure.com/"
azuredoc_apikey = "8a3892390cde4a448ccc80ca6cc02157"
loader1 = AzureAIDocumentIntelligenceLoader(
    api_endpoint=azuredoc_endpoint,
    api_key=azuredoc_apikey,
    file_path=file_path,
    api_model="prebuilt-layout",
)

# RAG 架構
# 選擇 splitter 並將文字切分成多個 chunk
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = loader1.load_and_split(splitter)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

chat_history = []
input_text = input(">>> ")
while input_text.lower() != "bye":
    if input_text:
        response = agent_executor.invoke(
            {
                "input": input_text,
                "chat_history": chat_history,
            }
        )
        chat_history.extend(
            [
                HumanMessage(content=input_text),
                AIMessage(content=response["output"]),
            ]
        )
        # chat_history.append((query, result["answer"]))
        print(response["output"])
