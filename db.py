from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    AzureAIDocumentIntelligenceLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["MONGO_URI"] = os.getenv("mongo_uri")
os.environ["AZUREDOC_ENDPOINT"] = os.getenv("azuredoc_endpoint")
os.environ["AZUREDOC_APIKEY"] = os.getenv("azuredoc_apikey")


def check_rule_by_supplier_name(collection, supplier_name):
    query = {"supplier_name": supplier_name}
    results = collection.find(query)
    rules_list = []
    for result in results:
        rules_list.append(
            result.get("rule", [])
        )  # 取得"rules"欄位，如果不存在則回傳null list
    return rules_list


def connect_to_mongo(uri, db_name, collection_name):
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    return collection


mongo_uri = os.environ["mongo_uri"]
database_name = "test"
collection_name = "rules"
collection = connect_to_mongo(mongo_uri, database_name, collection_name)

# Azure AI Document Intelligence 認證資訊 (請替換為您的資訊)
azuredoc_endpoint = os.environ["azuredoc_endpoint"]
azuredoc_apikey = os.environ["azuredoc_apikey"]
file_path = "./doc/上誼3.docx"

# 使用 Azure AI Document Intelligence Loader 載入文件
loader1 = AzureAIDocumentIntelligenceLoader(
    api_endpoint=azuredoc_endpoint,
    api_key=azuredoc_apikey,
    file_path=file_path,
    api_model="prebuilt-layout",
)
documents = loader1.load()

# 分割文件為較小的 chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = loader1.load_and_split(splitter)

# 建立向量資料庫
embeddings = OllamaEmbeddings(model="llama3.1:latest")
vectorstore = Chroma.from_documents(texts, embeddings)

# 建立 LLMChain
llm = ChatOllama(model="llama3.1:latest")
prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="請你根據提供的檔案回答出版商是哪間公司，請直接回答答案，不要口語化回答。不要有幻覺，不確定的資訊請直接說不知道。\n檔案內容：{context}",
)
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# 提出問題並獲取答案
question = "出版商是哪間公司？"
new_supplier = llm_chain.run(
    question=question, context=texts[0].page_content
)  # 使用第一個 chunk 作為 context

# 印出結果
print(new_supplier)

suppliers = check_rule_by_supplier_name(collection, new_supplier)
if suppliers:
    print(f"找到已存在的供應商資料:")
    for rule in suppliers:
        print(rule)  # 這裡會印出每個 document 的 "rules" 欄位
else:
    print(f"這是新的供應商表格！")
