from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


doc = ["What is the capital of India?","we are learning about embeddings in langchain","Embeddings are useful for semantic search and clustering"]

res = embeddings.embed_documents(doc)

print(res)