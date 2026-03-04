import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path="docs"):
    print(f"Loading documents from the '{docs_path}' directory...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory '{docs_path}' does not exist. Please create it and add your documents."
        )

    # Load documents from the "data" directory. Some files may not be UTF-8;
    # pass loader kwargs to handle alternate encodings or decoding errors.
    loader = DirectoryLoader(
        docs_path,
        glob="**/*.txt",
        show_progress=True,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "latin-1"},
    )
    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(
            f"No documents found in the '{docs_path}' directory. Please add some .txt files to process."
        )

    # for i, doc in enumerate(documents, start=1):
    #     # src = doc.metadata.get("source") if hasattr(doc, "metadata") else None
    #     # print(f"Document {i}: {src} (length: {len(doc.page_content)} characters)")
    #     print(f"\nDocument {i}:")  # Print the first 200 characters
    #     print(
    #         f" Source: {doc.metadata.get('source') if hasattr(doc, 'metadata') else 'N/A'}"
    #     )
    #     print(f" ContentLength: {len(doc.page_content)} characters")
    #     print(
    #         f" Content Preview: {doc.page_content[:200]}..."
    #     )  # Print the first 200 characters
    #     print(f" Metadata: {doc.metadata if hasattr(doc, 'metadata') else 'N/A'}")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    print(
        f"\nSplitting documents into chunks (chunk size: {chunk_size}, overlap: {chunk_overlap})..."
    )
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(documents)

    return split_docs

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Creating embeddings and storeing in Chroma vector database...")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    print("---> Storing embeddings in Chroma vector database...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=persist_directory,
    )
    print("--- Finished creating vector store ---")
    return vector_store

def main():
    # Load documents from the "data" directory
    documents = load_documents(docs_path="docs")
    # Split documents into chunks
    split_docs = split_documents(documents, chunk_size=1000, chunk_overlap=0)
    # Create vector store and persist to disk
    vector_store = create_vector_store(split_docs, persist_directory="db/chroma_db")


if __name__ == "__main__":
    main()
