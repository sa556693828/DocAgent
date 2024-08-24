loader = CSVLoader(
    file_path="./doc/標準格式.csv",
    encoding="utf-8",
)
data = loader.load()
docLoader = Docx2txtLoader("./doc/上誼.docx")
docData = docLoader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = text_splitter.split_documents(docData)
# print("文档分割结果:")
# for i, split in enumerate(splits, 1):
#     print(f"片段 {i}:")
#     print(f"{split.page_content[:50]}...")  # 只显示前50个字符
#     print()
embedding = OllamaEmbeddings(
    model="llama3.1",
)
vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
# vectorstore = Chroma.from_documents(documents, embedding)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
# result = retriever.invoke("這本書的書名是什麼？")
# print(result)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# template = """Use the following pieces of context to answer the question at the end.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Use three sentences maximum and keep the answer as concise as possible.
# Always say "thanks for asking!" at the end of the answer.

# {context}

# Question: {question}

# Helpful Answer:"""

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


# RAG chain
# question_answer_chain = create_stuff_documents_chain(gpt35Model, RAG_prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)
# )
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | RAG_prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is Task Decomposition?")

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
