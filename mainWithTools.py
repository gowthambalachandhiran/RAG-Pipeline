import os
from dotenv import load_dotenv
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from operator import itemgetter
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.runnables import RunnableLambda
from vector import get_retriever  # Custom retriever for internal review data

# --- Load Environment Variables ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables or .env file.")
if not serper_api_key:
    raise ValueError("SERPER_API_KEY not found in environment variables or .env file.")
# --- Initialize Groq LLM ---
llm = ChatGroq(
    temperature=0.7,
    model_name="llama3-70b-8192",
    groq_api_key=groq_api_key
)

# --- Initialize Retriever & Memory ---
retriever = get_retriever(source="csv")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)




# --- Initialize Serper Search Tool ---
serper_tool = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)


# --- Prompt Template ---
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are an expert in answering questions about a pizza restaurant. "
        "You have access to both internal reviews and public web search results for up-to-date information."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template(
        """
        Here are some relevant reviews:
        {reviews}

        Here is some information from Serper Search:
        {serper_search}

        Here is the question to answer:
        {question}
        """
    )
])



# --- Serper Search Wrapper ---
def serper_search_tool(query: str) -> str:
    """Use Serper to get up-to-date information."""
    try:
        return serper_tool.run(query)
    except Exception as e:
        return f"Serper error: {e}"

# --- Context Chain ---
context_retriever_chain = (
    {
        "reviews": itemgetter("question") | retriever,
        "serper_search": itemgetter("question") | RunnableLambda(serper_search_tool),
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history")
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --- Main Chat Loop ---
print("🍕 Welcome to the Pizza Restaurant QA Bot with Memory and DuckDuckGo Search!")
print("Type 'q' to quit at any time.")

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question: ")
    if question.lower() == "q":
        print("Exiting. Goodbye!")
        break

    try:
        current_chat_history = memory.load_memory_variables({})["chat_history"]

        print(f"Retrieving relevant reviews and Serpent Search results for: '{question}'...")

        response = context_retriever_chain.invoke(
            {"question": question, "chat_history": current_chat_history}
        )

        memory.save_context(
            {"input": question},
            {"output": response}
        )

        print("\nGroq LLaMA-3.3-70B Response:\n")
        print(response)
        print("\n\n")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your API keys are correct and 'vector.py' is properly set up.")
