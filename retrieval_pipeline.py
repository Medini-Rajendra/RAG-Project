from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

# query = "What percentage of the brain's neurons are located in the cerebellum, and why does this make it a challenge to simulate?"
query = "What is the capital of India?"

retriever = db.as_retriever(search_kwargs={"k": 3})

relevant_docs = retriever.invoke(query)

print(" --- Context ----")
for i, doc in enumerate(relevant_docs, start=1):
    print(f"Document {i}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("-" * 50)

# combine the query and relevant document contents
combined_input = f"""Based on the following documents, please answer the question: {query}

Documents:
{chr(10).join([doc.page_content for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say you don't know.
"""
# create a chatopenAI model
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

model = ChatOpenAI(model="gpt-4o")

# define the messages
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# invoke the model
result = model.invoke(messages)
print("\n --- Final Answer ---")
print(result.content)
