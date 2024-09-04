import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from datetime import datetime
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from pymongo import MongoClient
import certifi
from langchain_community.document_loaders import Docx2txtLoader, CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
import json

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "marketing"
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["MONGO_URI"] = os.getenv("MONGO_URI")

mongoUrl = os.environ["MONGO_URI"]

# Load the LLaMA model
llama3Model = ChatOllama(model="llama3.1:latest")
gpt4oModel = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
scoringModel = ChatOllama(model="scoringmodel")


# Tools
@tool
def get_themes_tags() -> list:
    """Get all the tags from tags_collection in MongoDB."""
    client = MongoClient(mongoUrl, tlsCAFile=certifi.where())
    db = client["prompt"]
    collection = db["tags_collection"]
    query = {}
    results = collection.find(query)
    return [result for result in results]


@tool
def articles_to_tags(themeWithTags: list, file_path: str) -> str:
    """
    analyze the article from all the tags in the theme
    """
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)

    if file_path.endswith(".docx"):
        userFile = Docx2txtLoader(file_path)
    elif file_path.endswith(".pdf"):
        userFile = PyPDFLoader(file_path)
    else:
        raise ValueError("Unsupported file format")

    userFileData = userFile.load()
    chunks = text_splitter.split_documents(userFileData)
    PromptTags = []
    for theme in themeWithTags:
        system_prompt = f"""
        你是一個retriever助理，請根據user提供的貼文，從以下theme+tags中找出符合此篇貼文的tags：
        theme: {theme["theme"]}
        tags: {theme["tags"]}
        -
        請按照以下格式返回JSON：
        theme: {theme["theme"]}
        tags: 符合的tags
        """
        retrieve_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{chunks}"),
            ]
        )
        retrieve_chain = retrieve_prompt_template | llm | StrOutputParser()
        output = retrieve_chain.invoke(
            {"chunks": chunks},
        )
        PromptTags.append(output)

    system_prompt = f"""
        你是一個retriever助理，請根據user提供的貼文，分析此貼文的writing style，並產生tags
        -
        請按照以下格式返回JSON：
        writingStyle: [產生的tags]
        """

    retrieve_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{chunks}"),
        ]
    )
    retrieve_chain = retrieve_prompt_template | llm | StrOutputParser()
    writingStyle_output = retrieve_chain.invoke(
        {"chunks": chunks},
    )
    final_output = json.dumps(
        {"writingStyle": writingStyle_output, "PromptTags": PromptTags}
    )
    return final_output


@tool
def insert_theme_tags(article: str, writingStyle: list, PromptTags: list, scores: dict):
    """Insert article with score and tags into MongoDB"""
    client = MongoClient(mongoUrl, tlsCAFile=certifi.where())
    db = client["article"]
    collection = db["articles"]
    # Create a list of documents to insert
    document = {
        "article": article,
        "writingStyle": writingStyle,
        "scores": scores,
        "PromptTags": PromptTags,
    }

    # # Insert multiple documents into the collection
    result = collection.insert_one(document)

    return f"Prompts saved with IDs: {result.inserted_id}"


# @tool
def scoring_article(file_path: str) -> dict:
    """scoring given Travel article with the Scoring Criteria , and return the scores"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)

    if file_path.endswith(".docx"):
        userFile = Docx2txtLoader(file_path)
    elif file_path.endswith(".pdf"):
        userFile = PyPDFLoader(file_path)
    else:
        raise ValueError("Unsupported file format")

    userFileData = userFile.load()
    chunks = text_splitter.split_documents(userFileData)

    criteria = """
        Scoring Criteria for Travel article
        1. SEO (20 points)
        16-20 points: The article uses appropriate keywords, has a clear structure, includes internal and external links, and has optimized meta titles and descriptions, effectively improving search engine rankings.
        8-15 points: Basically uses keywords, has an acceptable structure, but lacks internal or external links, and the meta title and description are not attractive enough.
        0-7 points: Almost no use of keywords, chaotic structure, lack of links, unable to effectively improve rankings.
        
        2. Content Quality (20 points)
        16-20 points: The content is in-depth and comprehensive, covering various aspects of information, able to arouse readers' interest and provide unique insights.
        8-15 points: The content is basically complete, but some aspects of the description are not detailed enough, failing to fully attract readers' attention.
        0-7 points: The content is superficial, lacking specific details or important background information, unable to arouse readers' interest.
        
        3. Practicality (15 points)
        12-15 points: Provides a wealth of practical information, such as specific suggestions, steps, and resources, which is of great help to readers' needs.
        6-11 points: Provides some practical information, but it is not comprehensive or detailed enough, and readers may need to find additional information.
        0-5 points: Lacks practical information, or the information provided is inaccurate, providing little help to readers' needs.
        
        4. Interactivity (15 points)
        12-15 points: The article encourages reader interaction, provides opportunities for questions or comments, and actively responds to reader feedback.
        6-11 points: Has a certain degree of interactivity, but lacks proactivity or has a lower frequency of interaction.
        0-5 points: Lacks interactivity, does not provide opportunities for readers to participate, or does not respond to reader feedback.
        
        5. Ease of Reposting (10 points)
        8-10 points: The content is highly shareable, providing unique insights and interesting stories that encourage readers to repost it on other platforms. The article has a clear structure, is easy to understand, and features engaging titles and subheadings that capture the reader's attention.
        4-7 points: The content is shareable but may lack sufficient appeal or uniqueness to fully inspire readers to repost. The article structure is acceptable, but some parts may not be clear or engaging enough.
        0-3 points: The content lacks appeal and fails to capture the reader's interest, with a chaotic structure or unclear expression, leading readers to be unwilling to repost.
        
        6. Ease of Collecting (10 points)
        8-10 points: The article contains practical and valuable content, providing specific suggestions or insights that make readers feel it is worth collecting. The content is presented clearly and is easy to understand, with guiding language that encourages readers to save it for future reference.
        4-7 points: The content has some value but may not be specific or in-depth enough to fully inspire readers to collect it. The presentation of the article is acceptable but lacks clear prompts or guidance, making it difficult for readers to decide whether to collect it.
        0-3 points: The content lacks practicality or value, fails to attract the reader's attention, and is poorly presented, leading readers to be unwilling to collect it.
        
        7. Ease of Liking (10 points)
        8-10 points: The article's content is engaging, featuring emotional resonance or interesting elements that strongly resonate with readers, encouraging them to like it. The conclusion encourages readers to express their appreciation for the content and provides clear prompts for liking.
        4-7 points: The content is acceptable but may lack sufficient emotional resonance or interest to fully inspire readers to like it. The prompts for liking are not prominent enough, and readers may need additional encouragement to do so.
        0-3 points: The content fails to evoke emotional resonance or interest in readers, is expressed blandly, and lacks prompts to encourage liking, leading readers to be unwilling to like it.
    """
    # 寫作風格應該定為tag，評分評價
    system_prompt = f"""
        Please score the travel article given below with this Scoring Criteria. You have to analyze step by step.
        Scoring Criteria: {criteria}
        article: {chunks}
        -
        Please return the result in the following format as JSON:
        SEO: score,
        ContentQuality: score,
        Practicality: score,
        Interactivity: score,
        Reposting: score,
        Collecting: score,
        Liking: score,
        total_score: user total score,
    """
    response = gpt4oModel.invoke(system_prompt)
    return response


# 工具箱
tools = [get_themes_tags, articles_to_tags, scoring_article, insert_theme_tags]

# Define the system prompt
system_prompt = "You are an assistant that completes user needs."

# Create the agent
graph = create_react_agent(gpt4oModel, tools, state_modifier=system_prompt)

inputs = {
    "messages": [
        (
            "user",
            """
            請先抓取資料庫中現有的theme，以及theme所擁有的tags。
            找到後，
            1. 分析此貼文有包含的tags
            2. 分析此貼文的writing style
            3. 評分此貼文，並得到scores
            4. 將資料插入到資料庫中
            """,
        ),
        (
            "human",
            """旅遊貼文：./doc/tokyo2.docx""",
        ),
    ]
}

# Stream the response
for s in graph.stream(inputs, stream_mode="values"):
    message = s["messages"][-1]
    if isinstance(message, tuple):
        print(message)
    else:
        message.pretty_print()
