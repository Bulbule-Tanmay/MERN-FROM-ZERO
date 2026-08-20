from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Documents
DOCUMENTS = [
    "Virat Kohli is an Indian international cricketer who captained India and is known for his aggressive batting.",
    "Mahendra Singh Dhoni, affectionately known as MS Dhoni, is a former Indian cricket captain and wicketkeeper-batsman.",
    "Sachin Tendulkar is a legendary Indian batsman and is widely regarded as one of the greatest cricketers in history.",
]


def get_top_documents(query, embeddings, documents, top_k=3):
    """Return the most relevant document indexes for the user query."""
    query_embedding = embeddings.embed_query(query)
    document_embeddings = embeddings.embed_documents(documents)
    cosine_similarities = cosine_similarity([query_embedding], document_embeddings).flatten()
    return cosine_similarities.argsort()[-top_k:][::-1]


def answer_query(query, llm, embeddings, documents, top_k=3):
    """Answer a question using the most relevant documents as context."""
    if not query or not query.strip():
        return "Please enter a valid prompt."

    relevant_indexes = get_top_documents(query, embeddings, documents, top_k=top_k)
    context_docs = "\n\n".join([documents[i] for i in relevant_indexes])

    response = llm.invoke([
        SystemMessage(content=(
            "You are a helpful assistant. Use only the context below to answer. "
            "If the answer is not in the context, say you do not have enough information.\n\n"
            f"Context:\n{context_docs}"
        )),
        HumanMessage(content=query),
    ])

    return response.content


def main():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model="smollm2:135m", temperature=0)

    print("Type 'exit' to quit.")

    while True:
        prompt = input("Enter your prompt: ").strip()
        if prompt.lower() == "exit":
            print("Goodbye!")
            break

        answer = answer_query(prompt, llm, embeddings, DOCUMENTS, top_k=3)
        print("\nAnswer:")
        print(answer)
        print("-" * 40)


if __name__ == "__main__":
    main()
