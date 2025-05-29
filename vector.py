from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch  # Required for HuggingFace embeddings

csv_path = "./Data/realistic_restaurant_reviews.csv"
db_location = "./chrome_langchain_db"

# Load CSV data
df = pd.read_csv(csv_path, delimiter=",", quotechar='"')


# Use Sentence Transformers embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- CSV retriever setup ---
add_documents_csv = not os.path.exists(db_location)
if add_documents_csv:
    documents_csv = []
    ids_csv = []
    for i, row in df.iterrows():
        document = Document(
            page_content=f"{row['Title']} {row['Review']}",
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        documents_csv.append(document)
        ids_csv.append(str(i))

vector_store_csv = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents_csv:
    vector_store_csv.add_documents(documents=documents_csv, ids=ids_csv)

retriever_csv = vector_store_csv.as_retriever(search_kwargs={"k": 5})

# --- PDF retriever setup with chunking and similarity options ---

pizza_pdf_path = "./Data/pizza.pdf"
pizza_db_location = "./chroma_pizza_db"

add_documents_pdf = not os.path.exists(pizza_db_location)
if add_documents_pdf:
    loader = PyPDFLoader(pizza_pdf_path)
    documents_pdf = loader.load()
    # Chunking the PDF documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs_pdf = text_splitter.split_documents(documents_pdf)
    ids_pdf = [str(i) for i in range(len(chunked_docs_pdf))]
else:
    chunked_docs_pdf = None
    ids_pdf = None

vector_store_pdf = Chroma(
    collection_name="pizza_pdf",
    persist_directory=pizza_db_location,
    embedding_function=embeddings
)

if add_documents_pdf and chunked_docs_pdf:
    vector_store_pdf.add_documents(documents=chunked_docs_pdf, ids=ids_pdf)

# Similarity search options for PDF retriever
retriever_pdf = vector_store_pdf.as_retriever(
    search_type="cosine",  # You can change to "mmr" or other supported types
    search_kwargs={"k": 5}
)

# --- Retriever selection function ---
def get_retriever(source="csv"):
    if source == "pdf":
        return retriever_pdf
    return retriever_csv

if __name__ == "__main__":
    query = "Talk about the pizza quality and service and options for vegan pizza."    # Example: Use CSV retriever
    results = retriever_csv.get_relevant_documents(query)
    print("CSV Results:")
    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(doc.page_content)
        print(f"Metadata: {doc.metadata}")
        print("-" * 40)

    # Example: Use PDF retriever
    results_pdf = retriever_pdf.get_relevant_documents(query)
    print("PDF Results:")
    for i, doc in enumerate(results_pdf, 1):
        print(f"Result {i}:")
        print(doc.page_content)
        print(f"Metadata: {doc.metadata}")
        print("-" * 40)