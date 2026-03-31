from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

load_dotenv()

# Step 1: Load Document
loader = PyPDFLoader("sample_resume.pdf")
documents = loader.load()

# Step 2: Chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# Step 3: Embeddings
embeddings = OpenAIEmbeddings()

# Step 4: Vector Store
vectorstore = FAISS.from_documents(chunks, embeddings)

# Step 5: Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k":2})

# Step 6: Query
query = input("Ask a question about the document")

relevant_docs = retriever.invoke(query)

context = "\n".join([doc.page_content for doc in relevant_docs])

# Step 7: LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)

print(response.content)