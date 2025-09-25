from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

print(conversation.predict(input="Hi, I am Prasad."))
print(conversation.predict(input="What's my name?"))
