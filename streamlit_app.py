import streamlit as st
import openai
import pinecone
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone

st.set_page_config(page_title="Tax Pro 2022", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Chat with the Tax Pro 2022, powered by GPT3.5")

# Sidebar for entering OpenAI key
with st.sidebar:
    st.title('OpenAI key')
    if 'openai_key' in st.secrets:
        st.success('OpenAI key already provided!', icon='✅')
        openai_key = st.secrets['openai_key']
    else:
        openai_key = st.text_input('Enter OpenAI key:', type='password')
        if not openai_key:
            st.warning('Please enter your OpenAI key!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')

# Store chat messages, and initialize the chat message history
if 'messages' not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question about the 2022 tax filing!"}]

# Display the prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if prompt := st.chat_input(disabled=not openai_key):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

openai.api_key = openai_key

# Function to get the GPT3.5's response
def get_assistant_response(messages):
    r = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
    )
    response = r.choices[0].message.content
    return response

# Pinecone key, environment, and index name used in the notebok
pinecone_key = st.secrets['pinecone_key']
pinecone_environment = st.secrets['pinecone_environment']
index_name = st.secrets['index_name']

# Connect to pinecone database
pinecone.init(api_key=pinecone_key, environment=pinecone_environment)
index = pinecone.Index(index_name)

# Set embedding model and vectorstore, need to be used for prompt embedding and information retrieval from pinecone database
embed_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = Pinecone(index, embed_model.embed_query, 'text')

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Retrieve similar prompts from pinecone database
            prompt_retrived_content = vectorstore.similarity_search(st.session_state.messages[-1]["content"], k=3)
            # Concatenate the retrieved prompts
            new_prompt = st.session_state.messages[-1]["content"]
            for document in prompt_retrived_content:
                new_prompt = new_prompt+". "+str(document).split("page_content='")[1].split("', metadata=")[0].replace('\n','')
            # Replace the original prompt with the concatenated prompts
            new_messages = st.session_state.messages.copy()
            new_messages[-1]["content"] = new_prompt
            # Get the GPT3.5's response
            response = get_assistant_response(new_messages)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message) # Add response to message history
