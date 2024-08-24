import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import Docx2txtLoader, CSVLoader
from langgraph.prebuilt import create_react_agent

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")


# LLM
claudeModel = ChatAnthropic(model_name="claude-3-sonnet-20240229")
gpt35Model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# ollamaModel = Ollama(model="llama2")


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


tools = [
    compare_files_with_llm,
]

system_prompt = """
    根據用戶資料中的出版商名稱查詢建檔規則，
    如果有查到就只回傳建檔規則不要加任何文字，
    如果沒有就比較用戶提供的文件與標準格式生成轉換規則，
    並且回傳轉換規則
"""
graph = create_react_agent(gpt35Model, tools, state_modifier=system_prompt)
inputs = {
    "messages": [("user", "此為新書資料")] + [("human", "file_path: ./doc/九歌.docx")]
}

for s in graph.stream(inputs, stream_mode="values"):
    message = s["messages"][-1]
    if isinstance(message, tuple):
        print(message)
    else:
        message.pretty_print()
