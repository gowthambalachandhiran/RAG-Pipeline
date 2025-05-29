import os
from dotenv import load_dotenv
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from operator import itemgetter

from vector import get_retriever # Your existing retriever setup

# Load environment variables from .env file
load_dotenv()

# Initialize the Groq client (using LangChain's ChatGroq wrapper)
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables or .env file.")

# Initialize ChatGroq LLM
llm = ChatGroq(
    temperature=0.7,
    model_name="llama3-70b-8192",
    groq_api_key=groq_api_key
)

# Initialize the retriever
retriever = get_retriever(source="pdf")

# Initialize conversational memory
# 'memory_key' is the key under which the chat history will be stored in the memory variables
# 'return_messages=True' means the history will be returned as a list of LangChain message objects
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- CORRECTED PROMPT TEMPLATE ---
# Use MessagesPlaceholder for chat history
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are an expert in answering questions about a pizza restaurant."
    ),
    MessagesPlaceholder(variable_name="chat_history"), # This is where the history goes
    HumanMessagePromptTemplate.from_template(
        """
        Here are some relevant reviews:
        {reviews}

        Here is the question to answer:
        {question}
        """
    )
])
# --- END CORRECTED PROMPT TEMPLATE ---

# Define the RAG chain with memory
context_retriever_chain = (
    {
        # Retrieve reviews based on the current question
        "reviews": itemgetter("question") | retriever, 
        # Pass the original question through
        "question": itemgetter("question"), 
        # Pass the chat history through (it will be a list of messages)
        "chat_history": itemgetter("chat_history") 
    }
    | prompt # Format the prompt with all inputs
    | llm # Invoke the LLM
    | StrOutputParser() # Parse the LLM's output to a string
)

print("Welcome to the Pizza Restaurant QA Bot with Memory!")
print("Type 'q' to quit at any time.")

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question: ")
    if question.lower() == "q":
        print("Exiting. Goodbye!")
        break

    try:
        # Load current chat history from memory
        # memory.load_memory_variables({}) returns a dict like {'chat_history': [HumanMessage(...), AIMessage(...)]}
        current_chat_history = memory.load_memory_variables({})["chat_history"]
        
        print(f"Retrieving relevant reviews and generating response for: '{question}'...")
        
        # Invoke the chain with the current question and chat history
        response = context_retriever_chain.invoke(
            {"question": question, "chat_history": current_chat_history}
        )

        # Save the current conversation turn to memory
        memory.save_context(
            {"input": question},
            {"output": response}
        )

        print("\nGroq LLaMA-3.3-70B Response:\n")
        print(response)
        print("\n\n")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your Groq API key is correct and 'vector.py' is properly set up.")

