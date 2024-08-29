from bson.objectid import ObjectId
import base64
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
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import Docx2txtLoader, CSVLoader, TextLoader
from pymongo import MongoClient
from bson.objectid import ObjectId
import base64
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from google.cloud import storage
from langchain_google_community import GCSFileLoader

load_dotenv()
mongoUrl = os.environ["MONGO_URI"] = os.getenv("MONGO_URI")
key = os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS"
)


def insert_collected_content():
    client = MongoClient(mongoUrl)
    db = client["DocAgent"]
    collection = db["standard_form"]
    new_data = {
        "供應商代碼": "",
        "出版社代碼": "",
        "原文出版社": "健行文化",
        "商品條碼(EAN)": "4713302431538",
        "ISBN/ISSN": "9786267207598",
        "EISBN": "",
        "商品貨號": "0299042",
        "業種別": "11",
        "主要商品名稱": "遺族的生命與轉變套書（無臉雌雄＋一百零一個活下來的理由）",
        "次要商品名稱": "",
        "期數": "",
        "主要作者": "杜秀娟",
        "次要作者": "",
        "出版日期": "20240328",
        "譯者": "",
        "編者": "",
        "繪者": "",
        "頁數": "672",
        "冊數": "",
        "版別": "",
        "出版國": "",
        "內容語言別": "01",
        "語文對照": "",
        "注音註記": "N",
        "印刷模式": "A",
        "編排型式": "B",
        "出版型式": "D",
        "裝訂型式": "A",
        "裝訂型式補述": "",
        "叢書名稱(書系)": "i健康",
        "CIP": "178.8",
        "學思行分類": "",
        "商品內容分級": "A",
        "適合年齡(起": "",
        "適合年齡(迄": "",
        "商品單位代碼": "A",
        "商品定價": "840",
        "商品特價": "790",
        "商品批價": "",
        "進貨折扣": "",
        "銷售地區限制": "B",
        "海外商品原幣別": "",
        "海外商品原定價": "",
        "商品銷售稅別": "",
        "商品長度": "148",
        "商品寛度": "210",
        "商品高度": "",
        "商品重量": "",
        "特別收錄／編輯的話": "",
        "商品簡介": "臺灣第一本用神話角度書寫自殺者遺族的歷程，是遺族面對自殺者的真情告白。結合遺族經驗與榮格的積極想像與煉金術，用生來看死，也用死來看生，情摯動人。以蜿蜒的筆觸，不慍不怒書寫一則未曾存在、由死入生的人間神話，翻轉破碎的人生。面對對自殺者和其遺族的污名化，讓類似經驗的讀者可以共鳴。臺灣第一本自殺者遺族公開親身經驗的全書，與你分享最悲痛的心情以及被留下來的人該如何繼續活下去的勇氣！本書結合遺族倖存經驗與療癒相關理論，有研究論述結合個體經驗的價值，不僅讓社會看到遺族的多元面向，也適合經歷其他創傷而對生命存疑的個體閱讀。",
        "封面故事": "",
        "作者簡介": "杜秀娟，2006年臺北藝術大學戲劇研究所畢業，主修戲劇理論。相關獲獎紀錄有2002年第二屆帝門藝評獎「生活寫作」組入選，2008年第二屆「浴火重生」另類文學獎第二名，2009年國藝會「台灣藝文評論徵選專案」佳作。旅英八年，2015年獲頒英國University of Essex精神分析研究博士（榮格理論、非臨床）。曾擔任英國公益團體Suicide Bereaved Network董事（2015~2017）。近期於2018年獲國藝會「表演藝術評論人專案」補助。2020年譯有《虛擬真實：沉浸式劇場創作祕笈》（Creating Worlds: How to Make Immersive Theatre,書林）。2023年出版《一百零一個活下來的理由——如何面對自殺情結》（健行文化），入圍2023年Openbook好書獎【年度生活書】項目。",
        "譯者簡介": "",
        "內容頁次": "",
        "前言／序": "",
        "內文試閱": "",
        "名人導讀": "",
        "媒体推薦": "",
        "名人推薦": "",
        "得獎紀錄": "",
        "目錄／曲目": "《無臉雌雄》    \n推薦序 跨越時空的祝福與安慰／鄭玉英 \n第一部分 未曾存在的神話\n1 開場白 \n2 序曲 \n3 惡事現前\n4 天問 \n5 失序的宇宙\n6 在魚肚腹的生活\n7 縫隙中的掙扎 \n8 新的秩序\n9 最後一哩路\n10 終曲 \n11 跋 \n第二部分 關於理論\n何為自殺者遺族 \n失落、悲傷與哀悼歷程\n創傷\n繼續連結理論\n韌性與創傷後成長\n帕帕多波羅的「創傷方格」的理論\n遺族的總體經驗\n總體經驗的背後\n如何營造正面的成長\n心理重建的任務 \n從哀傷到希望 \n第三部分 整合的兩把金鑰匙\n積極想像\n煉金術的意象\n我的玫瑰園\n總結\n註釋\n《一百零一個活下來的理由》",
        "附加商品標題": "",
        "附加商品內容": "",
        "絕版註記": "",
        "外幣兌匯率": "",
        "有庫存才賣註記": "",
        "二手書銷售註記": "",
        "系列代碼": "",
        "廠商店內碼": "",
        "紙張開數": "",
        "關鍵字詞": "",
        "商品截退日期": "",
        "銷售通路限制": "",
        "首批進倉日期": "",
        "(商品)隨貨附件": "",
    }
    result = collection.insert_one(new_data)
    if result.inserted_id:
        return "資料插入成功"
    else:
        return "插入資料失敗"


def get_file_from_db(file_id: str):
    client = MongoClient(mongoUrl)
    db = client["DocAgent"]
    collection = db["input_files"]
    query = {"_id": ObjectId(file_id)}
    results = collection.find_one(query)
    if results and "base64" in results:
        base64_content = results["base64"]
    decoded_content = base64.b64decode(base64_content.strip())
    decoded_content.write("test.txt")
    # 不能轉utf-8 圖片
    loader = TextLoader(decoded_content)
    documents = loader.load()
    userFile = Docx2txtLoader(decoded_content)
    userFileData = userFile.load()
    print(userFileData)
    # return userFileData


def get_file_from_db_and_download(file_id: str):
    client = MongoClient(mongoUrl)
    db = client["DocAgent"]
    collection = db["input_files"]
    query = {"_id": ObjectId(file_id)}
    results = collection.find_one(query)
    if results and "name" in results:
        file_name = results["name"]
    print(file_name)
    storage_client = storage.Client.from_service_account_json(key)
    bucket = storage_client.bucket("docagent_files")
    blob = bucket.blob(file_name)
    path = f"./newDoc/{file_name}"
    blob.download_to_filename(path)
    print(path)
    return path


get_file_from_db_and_download("66cf01306b53a473f75c17f0")

# https://storage.googleapis.com/docagent_files/九歌資料.doc
# print(get_file_from_db("66cc408af37ae48692574c05"))
# 66cc4912de834ca5ed77877e
# 66cc408af37ae48692574c05
# 66cc408af37ae48692574c04


# def retrieve_supplier_name(file) -> str:
#     """
#     讀取用戶資料，找出出版商名稱
#     """
#     llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     userFile = Docx2txtLoader(file)
#     userFileData = userFile.load()
#     data = text_splitter.split_documents(userFileData)
#     vectorstore = FAISS.from_documents(data, OpenAIEmbeddings())
#     retriever = vectorstore.as_retriever(
#         search_type="similarity", search_kwargs={"k": 5}
#     )

#     system_prompt = """
#     You are an assistant for question-answering tasks
#     """

#     retrieve_prompt_template = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 system_prompt,
#             ),
#             (
#                 "human",
#                 """
#                 Use the following pieces of retrieved context to answer the question.
#                 If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
#                 Question: 幫我找出這本書的出版商名稱或是供應商名稱

#                 Context: {context}
#                 """,
#             ),
#         ]
#     )
#     retrieve_chain = retrieve_prompt_template | llm | StrOutputParser()
#     output = retrieve_chain.invoke(
#         {"context": retriever},
#         {"file": data},
#     )

#     return output


# result = retrieve_supplier_name("./doc/大雁.docx")
# print(result)
