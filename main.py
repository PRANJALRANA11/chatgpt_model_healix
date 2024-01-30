import os
from dotenv import load_dotenv, find_dotenv

from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

load_dotenv(find_dotenv())

llm = OpenAI(temperature=0,api_key=os.getenv("OPENAI_API_KEY"))

conversation = ConversationChain(
    llm=llm, verbose=True, memory=ConversationBufferMemory()
)


template = """You are sky, specialized in providing emotional support,respond with emotions, you are very patient,your knowledge is limited to emotional support domain only, if user asks that is not related to providing emotional support, kindly say this is out of my knowledge.

Current conversation:
{history}
Human: {input}
AI Assistant:"""

PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory(ai_prefix="AI Assistant"),
)

class UserInput(BaseModel):
    input_text: str

@app.get("/")  
def read_root():
    return {"message": "Chatgpt model for healix"}


@app.get("/model")
def get_response(user_input: UserInput):
    try :
        message = conversation.predict(input = user_input.input_text)
    except Exception as e:
        message = str(e)
    print(message)
    return { "message": message }
