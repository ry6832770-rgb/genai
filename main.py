import os
import sys
import warnings
import streamlit as st

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

st.set_page_config(
    page_title="Statistics RAG Assistant",
    page_icon="📚",
    layout="wide",
)

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment variables
load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    raise RuntimeError("NVIDIA_API_KEY not found. Check your .env file.")

# 2. Document Loading
pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "STATS.pdf")

if not os.path.isfile(pdf_path):
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

print(f"Loading document from: {pdf_path}")

loader = PyPDFLoader(pdf_path)
docs = loader.load()

print(f"Loaded {len(docs)} document pages.")


# 3. Document Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)

chunks = text_splitter.split_documents(docs)

print(f"Created {len(chunks)} text chunks.")


# 4. Embeddings
print("Creating embeddings...")

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# 5. Qdrant Vector Database
print("Creating Qdrant vector store...")

client = QdrantClient(location=":memory:")

client.create_collection(
    collection_name="rag_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

vectorstore = QdrantVectorStore(
    client=client, collection_name="rag_collection", embedding=embeddings
)

vectorstore.add_documents(chunks)

print("Indexing complete!")


# 6. Initialize NVIDIA LLM
print("Initializing NVIDIA LLM...")

llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    nvidia_api_key=api_key,
    temperature=0.2,
    max_completion_tokens=2500,
)

print("NVIDIA LLM initialized!")


# 7. Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# 8. RAG Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert tutor in Statistical Techniques. "
            "Answer the question accurately based ONLY on the provided context. "
            "If the answer cannot be found in the context, say that you don't know.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


# 9. RAG Chain
rag_chain = prompt | llm | StrOutputParser()


# 10. Query Function
def query_rag(question: str):

    print("\n" + "=" * 60)
    print(f"Query: {question}")
    print("=" * 60)

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)

    print(f"\nRetrieved {len(retrieved_docs)} relevant context chunks:")

    for i, doc in enumerate(retrieved_docs, start=1):

        page_num = doc.metadata.get("page", "Unknown")

        preview = doc.page_content.strip().replace("\n", " ")[:150]

        print(f"[{i}] (Page {page_num}): {preview}...")

    if not retrieved_docs:
        print("No relevant documents found.")
        return None

    # Build context
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Generate response
    print("\nGenerating response from NVIDIA LLM...")

    response = rag_chain.invoke({"context": context, "question": question})

    print("\nAnswer:")
    print(response)

    print("=" * 60)

    return response


# 11. Streamlit User Interface

st.title("📚 Statistics RAG Assistant")

st.write(
    "Ask any question related to Statistical Techniques " "from your uploaded PDF."
)

st.divider()

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is the principle of least squares?",
)

if st.button("Get Answer", type="primary"):

    if question.strip():

        with st.spinner("Searching PDF and generating answer..."):

            try:
                answer = query_rag(question)

                st.success("Answer generated successfully!")

                st.subheader("Your Answer")
                st.write(answer)

            except Exception as e:
                st.error(f"Error: {e}")

    else:
        st.warning("Please enter a question.")
