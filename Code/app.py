import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from utils import user_input, get_pdf_text_from_directory, get_text_chunks, get_vector_store, get_conversational_chain

st.set_page_config(page_title="ITS", page_icon='📃')

st.header("Python Intelligent Tutoring System ✏️📖")

# Preload PDF data
raw_text = get_pdf_text_from_directory()
text_chunks = get_text_chunks(raw_text)
get_vector_store(text_chunks)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        AIMessage(content="Hello there👋, I can help you today.")
    ]

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message('AI'):
            st.info(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# Accepting user input
user_question = st.chat_input("Ask you'r Question")
if user_question is not None and user_question != "":
    st.session_state.chat_history.append(HumanMessage(content=user_question))

    with st.chat_message("Human"):
        st.markdown(user_question)

    # Add a spinner for response generation
    with st.spinner("Thinking... Please wait."):
        # Process the user question
        response = user_input(user_question, st.session_state.chat_history)

        # Remove any unwanted prefixes from the response
        response = response.replace("AI response:", "").replace("chat response:", "").replace("bot response:", "").strip()

    with st.chat_message("AI"):
        st.write(response)

    st.session_state.chat_history.append(AIMessage(content=response))