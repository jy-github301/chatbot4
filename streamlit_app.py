import streamlit as st
import pinecone

pinecone_key="8ddf1bc3-23b1-41d1-89a7-4052bf264f7c"
pinecone_environment='gcp-starter'
index_name = 'tax-rag'

# Set up Pinecone

pinecone.init(api_key=pinecone_key, environment=pinecone_environment)
pinecone_index = pinecone.Index('tax-rag')

# Create a Streamlit widget for inputting vectors
input_vector = st.text_input("Enter a vector (comma-separated):")

# Convert the input to a list of floats
query_vector = [float(x.strip()) for x in input_vector.split(",")]

# Query Pinecone for similar vectors
if query_vector:
    results = pinecone_index.query(queries=[query_vector], top_k=10)
    for result in results:
        st.write(result)

# Add a button for adding vectors to Pinecone
add_button = st.button("Add vector")
if add_button:
    vector_input = st.text_input("Enter a vector to add (comma-separated):")
    vector = [float(x.strip()) for x in vector_input.split(",")]
    id_input = st.text_input("Enter an ID for the vector:")
    pinecone_index.add(vectors=[vector], ids=[id_input])

# Add a slider for controlling the number of results to display
num_results = st.slider("Number of results to display", min_value=1, max_value=100, value=10)
if query_vector:
    results = pinecone_index.query(queries=[query_vector], top_k=num_results)
    for result in results:
        st.write(result)
