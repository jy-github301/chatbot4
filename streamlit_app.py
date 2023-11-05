import streamlit as st
import openai
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


embed_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


pinecone.init(api_key=pinecone_key, environment=pinecone_environment)
index = pinecone.Index('tax-rag')
vectorstore = Pinecone(index, embed_model.embed_query, text_field = "text")



st.set_page_config(page_title="TaxPro", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("TaxPro")

# Sidebar for entering OpenAI key
with st.sidebar:
    st.title('OpenAI key')
    openai_key = st.text_input('Enter OpenAI key:', type='password')
    if not openai_key:
        st.warning('Please enter your OpenAI key!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')

 # Store chat messages, and initialize the chat message history

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Ask me any questions"}]

# Display the prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"]) 

# User-provided prompt
if prompt := st.chat_input(disabled=not openai_key):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Function to get the GPT3.5's response
def get_assistant_response(allmessages):

    llm=ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo",openai_api_key=OPENAI_API_KEY)

    chain = ConversationalRetrievalChain.from_llm(llm,
                                              vectorstore.as_retriever(search_kwargs={'k': 5}),
                                              return_source_documents=True)
         
    response = chain.result.message.content
    return response

if len(st.session_state.messages)>1:
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_assistant_response(st.session_state.messages) 
            st.write(response)

    newmessage = {"role": "assistant", "content": response}
    st.session_state.messages.append(newmessage)
