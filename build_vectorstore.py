import os
import pickle
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Configuration
DOCUMENTS_DIR = "documents"
VECTORSTORE_PATH = "vectorstore.pkl"

def load_documents():
    """Load all .txt files from the documents folder."""
    docs = []
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        print(f"Created '{DOCUMENTS_DIR}' folder. Add .txt files to it!")
        return docs

    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCUMENTS_DIR, filename)
            try:
                loader = TextLoader(filepath, encoding="utf-8")
                docs.extend(loader.load())
                print(f"Loaded: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return docs

def create_sample_document():
    """Create a fallback sample document if no files exist."""
    sample_text = (
        "The capital of France is Paris.\n"
        "Python is a programming language created by Guido van Rossum.\n"
        "Machine learning is a subset of artificial intelligence.\n"
        "Retrieval-Augmented Generation (RAG) improves LLMs by adding external knowledge."
    )
    return [Document(page_content=sample_text, metadata={"source": "sample.txt"})]

def main():
    print("Loading documents...")
    docs = load_documents()

    if not docs:
        print("No .txt files found in 'documents/' folder.")
        print("Adding sample document for demo...")
        docs = create_sample_document()

    print(f"Total documents loaded: {len(docs)}")

    # Split into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,  # Better overlap than 50
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    # Create embeddings
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build FAISS vector store
    print("Building FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save to disk
    with open(VECTORSTORE_PATH, "wb") as f:
        pickle.dump(vectorstore, f)

    print(f"Vector store saved to '{VECTORSTORE_PATH}'")
    print("Build complete! You can now run: streamlit run app.py")

if __name__ == "__main__":
    main()