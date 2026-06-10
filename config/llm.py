from langchain_groq import ChatGroq

from config.settings import settings


llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,

    model="llama-3.3-70b-versatile",

    temperature=0
)