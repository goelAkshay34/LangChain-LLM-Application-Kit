# LangChain
# getting started with installing library
# 1. pip install langchain
# 2. pip install langchain-openai

# Expose OpenAI API key via 
export OPENAI_API_KEY = "..."

from langchain_openai import ChatOpenAI
llm = ChatOpenAI()

# Prompt template - converts raw user input to a better input to LLM

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."), 
    ("user", "{input}")
])
# Combine these into simple LLM chain:
chain = prompt | llm

chain.invoke({"input": "how can langsmith help with testing?"})

#optput of this chain is a message. Better add 
# simple output parser to convert the chat message
# to a string

from langchain_core.output_parsers inport StrOutputParser
output_parser = StrOutputParser()

# Add this to a previous chain
chain = prompt | llm | output_parser

# Now invoking will now be a string (rather than ChatMessage)

chain.invoke({"input": "How can langsmith help with testing?"})

##############################################################################
# Retrieval Chain
# If you have too much data, then pass it to LLM directly.
# Use Retriver to fetch only the most relevant pieces & pass them in
# In this instance, we will populate a vector store & use them as a retrieval
# To load data: Use WebBaseLoader, which requires installing BeautifulSoup:
#  pip install beautifulsoup4

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()

# Next Step:
# We need to index it into a vectorstore
# This requires embedding model & a vectorstore

# We will use OpenAI LLM

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# We will use this embedding model to ingest documents into a vectorstore.
# We will use simple local vectorstore, FAISS, 
# pip install faiss-cpu

from langchain_community.vectorstores import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = faiss.from_documents(documents,embeddings)

# Now data is indexed in a vectorstore, we will create a retrieval chain.
# This chain will take an incoming question, lookup the relevant documents,
# then pass those docs along with original ques into an LLM & ask it to answer

from langchain.chains.combine_documents import create_stuff_documents_chain
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
                                          <context>
                                          {context}
                                          </context>
                                          Question: {input}""")
document_chain = create_stuff_documents_chain(llm,prompt)

from langchain.chains import create_retrieval_chain
retriever = vector.as_retriever()
retriever_chain = create_retrieval_chain(retriever, document_chain)

# we can now invoke this chain
# This returns a dictionary - the response from the LLM is in the answer key

response = retriever_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])

# We have now successfully set up a basic retriver chain.
# This answers single questions.
# We will turn this chain into one that can answer followup questions?

## Conversational Retrieval Chain
# 1. This will not just the recent input, but also take the whole history into account
# 2. The final LLM chain should likewise take the whole history into account

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

# Pass a prompt into LLM to generate this search query

prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="chat_history"),
                                           ("user","{input}"),
                                           ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conbersation")
                                           ])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# We will test this out by passing in an instance where the user is asking a follow up question.

from langchain_core.messages import HumanMessage, AIMessage

chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
retriever_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)
retriever_chain = create_retrieval_chain(retriever_chain, document_chain)

# We can now test this out end-to-end

chat_history = [HumanMessage(content="Can Lnagsmith help test my LLM application?"), AIMessage(content="Yes")]
retriever_chain.invoke({chat_history,
                        "input": "Tell me how"
                        })

#########################################################################################################################
## Agent

from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about Langsmith. For any qustions aboubt Langsmith, you must use this tool!",
)

# We will use search tool: Tavily - This requires API key

export TAVILY_API_KEY= ...

from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults()

# We can now create a list of tools we want to work with:

tools = [retriever_tool, search]

# We have the tools now, we will create an agent to use them
# pip install langchainhub

# Now we can use it to get a predefined prompt

from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent =  create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose = True)

# We can now invoke the agent. We can ask it questions about LangSmith:

agent_executor.invoke({"input": "how can langsmith help with testing?"})

# Ask something else

agent_executor.invoke({"input", "what is the weather in SF"})

# We can have conversations with it:
chat_history = [HumanMessage(content = "Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
agent_executor.invoke({
    "chat_history": chat_history,
    "input"= "Tell me how"
})

########################################################################################################################
# Serving with LangServe
# It helps developers deploy LanChain chains as a REST API

# pip install "langserve[all]"

# To create a server, we will make serve.py file. It will have three things:
# 1. The definition of our chain that we have built
# 2. Our FastAPI app
# 3. A definition of a route from which to serve the chain, which is done with 
#    langserve.add_routes

############################################################################################################################
# serve.py

from typing import List
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langserve import add_routes

# 1. Load Retriever
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
