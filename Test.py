import os
from dotenv import load_dotenv
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, trim_messages

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


model = ChatOpenAI(model="gpt-3.5-turbo")


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


config = {"configurable": {"session_id": "abc2"}}

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)


response = with_message_history.invoke(
    {"messages": [HumanMessage(content="hi! I'm todd")], "language": "Spanish"},
    config=config,
)

print(response.content)

# for r in with_message_history.stream(
#     {
#         "messages": [HumanMessage(content="hi! I'm todd. tell me a joke")],
#         "language": "English",
#     },
#     config=config,
# ):
#     print(r.content, end="|")

# trimmer = trim_messages(
#     max_tokens=65,
#     strategy="last",
#     token_counter=model,
#     include_system=True,
#     allow_partial=False,
#     start_on="human",
# )

# messages = [
#     SystemMessage(content="you're a good assistant"),
#     HumanMessage(content="hi! I'm bob"),
#     AIMessage(content="hi!"),
#     HumanMessage(content="I like vanilla ice cream"),
#     AIMessage(content="nice"),
#     HumanMessage(content="whats 2 + 2"),
#     AIMessage(content="4"),
#     HumanMessage(content="thanks"),
#     AIMessage(content="no problem!"),
#     HumanMessage(content="having fun?"),
#     AIMessage(content="yes!"),
# ]

# print(trimmer.invoke(messages))
