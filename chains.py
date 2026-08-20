import os

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)

prompt = PromptTemplate(
    template="Give me the top 5 points about {topic} in bullet points.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Give me short summary about {topic} in bullet points.",
    input_variables=["topic"]
)
parser = StrOutputParser()

chain = prompt | model | parser

output = chain.invoke({"topic": "LangChain"})

print(output)