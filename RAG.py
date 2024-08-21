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


template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is Task Decomposition?")
