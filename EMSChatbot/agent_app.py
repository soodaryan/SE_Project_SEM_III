import streamlit as st
from RAG_chat_bot import ChatBot

def chat(query) : 
    """Agent caller. Returns the ouput to Query by calling the RAG agent."""
    agent = ChatBot()
    print("done")
    resp = agent.query(query)
    print("done")
    return resp.response



# def show_ui(prompt_to_user="How may I help you?"):

#     """The function for Streamlit"""

#     if "messages" not in st.session_state.keys():
#         st.session_state.messages = [{"role": "You are an expert in data analysis and provide support to employees.", "content": prompt_to_user}]

#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.write(message["content"])

#     if prompt := st.chat_input():
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.write(prompt)

#     if st.session_state.messages[-1]["role"] != "assistant":
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 response = chat(prompt)
#                 st.markdown(response)
#         message = {"role": "assistant", "content": response}
#         st.session_state.messages.append(message)


def show_ui(prompt_to_user="How may I help you?"):

    """The function for Streamlit"""

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "You are an expert in data analysis and provide support to employees.", "content": prompt_to_user}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Validate the user input before sending to chat()
        if isinstance(prompt, str) and prompt.strip():
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat(prompt)  # Ensure prompt is a valid string
                    st.markdown(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
        else:
            st.error("Invalid input. Please enter a valid query.")


def run():
    
    """Wrapper Function to start UI"""

    show_ui("What would you like to know?")


st.title("Employee Help Chatbot")
run()