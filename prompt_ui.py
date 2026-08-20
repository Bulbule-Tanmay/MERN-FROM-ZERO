from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import streamlit as st




st.title("LangChain Ollama LLM Test")


llm = ChatOllama(model="tinyllama", temperature=0,base_url="http://localhost:11434"  )

paper_input = st.selectbox("Select a paper", ["attention all you need", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "RoBERTa: A Robustly Optimized BERT Pretraining Approach  "])
length_input = st.slider("Select the length of the summary", 1, 10, 5)
style_input = st.radio("Select the type of summary", ["mathematical", "number", "technical"])


temp = PromptTemplate(
    template= """Please summarize the research paper titled "{paper_input}" with the following
specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}
1. Mathematical Details:
- Include relevant mathematical equations if present in the paper.
- Explain the mathematical concepts using simple, intuitive code snippets
where applicable.
2. Analogies:
- Use relatable analogies to simplify complex ideas.
If certain information is not available in the paper, respond with: "Insufficient
information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and
length.""",
    variables=["paper_input", "style_input", "length_input"]
)


prompt = temp.invoke({"paper_input": paper_input, "style_input": style_input, "length_input": length_input})

if st.button("run"):
    result = llm.invoke(prompt)   
    st.write(result.content)