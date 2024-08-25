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
from pymongo import MongoClient
import pypandoc


load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")


# LLM
claudeModel = ChatAnthropic(model_name="claude-3-sonnet-20240229")
gpt35Model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# ollamaModel = Ollama(model="llama2")


def transform_data(file_path1: str, rule: str):
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


result = transform_data(
    "./doc/九歌資料.docx",
    """
1. 書名=>主要商品名稱
2. 書號=>商品貨號
3. 出版社=>原文出版社
4. 出版日=>出版日期
5. 作者=>主要作者
6. 定價=>商品定價
7. 外文書名=>次要商品名稱
8. 特價=>商品特價
9. 作者原名=>次要作者
10. 開本=>商品長度、商品寬度（長度：148mm、寬度：210mm）
11. 譯者=>譯者
12. 裝訂=>裝訂型式
13. ISBN=>ISBN/ISSN（需去除-符號）
14. CIP=>CIP
15. EAN=>商品條碼(EAN)（需去除-符號）
16. 內文頁數=>頁數
17. 建議類別=>業種別
18. 本書特色=>商品簡介
19. 目標讀者群=>適合年齡(起)、適合年齡(迄)（需根據內容填寫）
20. 作者簡介=>作者簡介
21. 內容簡介=>內容簡介
22. 得獎與推薦=>得獎紀錄、名人推薦
23. 附件1.目錄=>目錄／曲目
24. 附件2.推薦序=>名人導讀
25. 附件3.內文試閱=>內文試閱

""",
)
print(result)


# compare_files_with_llm("./doc/九歌.docx")
# update_rule_by_supplier_name("安徒生", "九歌出版社的轉換規則")

# tools = [
#     compare_files_with_llm,
# ]

# system_prompt = """


#     根據用戶資料中的出版商名稱查詢建檔規則，
#     如果有查到就只回傳建檔規則不要加任何文字，
#     如果沒有就比較用戶提供的文件與標準格式生成轉換規則，
#     並且回傳轉換規則
# """
# graph = create_react_agent(gpt35Model, tools, state_modifier=system_prompt)
# inputs = {
#     "messages": [("user", "此為新書資料")] + [("human", "file_path: ./doc/九歌.docx")]
# }

# for s in graph.stream(inputs, stream_mode="values"):
#     message = s["messages"][-1]
#     if isinstance(message, tuple):
#         print(message)
#     else:
#         message.pretty_print()
