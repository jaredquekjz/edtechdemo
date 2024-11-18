import json
import os
from pathlib import Path
from typing import Generator, Optional, cast
import streamlit as st
# import streamlit_antd_components as sac  # Commented out since it's not used
import streamlit.components.v1 as components
from rich import print as rich_print
import urllib.parse  
import tempfile
import pypandoc
import base64
from datetime import datetime
from openai import OpenAI
import time
from langchain_fireworks import ChatFireworks
import os

# Settings
# assets
ASSETS_DIR = Path("assets")
PANDAI_PNG = str(ASSETS_DIR / "pandai.jpeg")
PANDAI_AVATAR_PNG = str(ASSETS_DIR / "pandai.jpeg")
LOTUS_PNG = str(ASSETS_DIR / "lotus.png")
with open(ASSETS_DIR / "styles.css", "r") as f:
    STYLE_CSS = f.read()

DISCLAIMER = """
The finetuned models have been trained on Singapore Ministry of Education's EdTech PS and other curricular frameworks.
"""

# Update MODELS dictionary to include the Fireworks model
MODELS = {
    "Base GPT4o-mini": "gpt-4o-mini-2024-07-18",
    "Finetuned GPT4o-mini": "ft:gpt-4o-mini-2024-07-18:personal::ATSN5C3L",
    "Finetuned llama3.1-8b": "accounts/jaredquek-1b3158/models/pandaitest"
}

# Add constant to identify model types
MODEL_TYPES = {
    "Base GPT4o-mini": "openai",
    "Finetuned GPT4o-mini": "openai",
    "Finetuned llama3.1-8b": "fireworks"
}

# Initialize Fireworks client (add near the OpenAI client initialization)
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
fireworks_llm = ChatFireworks(
    model="accounts/jaredquek-1b3158/models/pandaitest",
    temperature=0.4,
    max_tokens=1000,
    api_key=FIREWORKS_API_KEY
)

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def display_chat_ui():
    # Display image and title
    st.set_page_config(page_title="EdTech Bot", page_icon=LOTUS_PNG)
    # Inject CSS to style page
    st.markdown(f"<style>{STYLE_CSS}</style>", unsafe_allow_html=True)

        # Create three columns with the middle one twice as wide
    col1, col2, col3 = st.columns([1, 5, 1])

    with col2:
        # Specify the image width to control its size
        st.image(PANDAI_PNG)
        st.title("EdTech Model Tester")

    # Page sidebar
    # Add model selection to sidebar
    selected_model_name = st.sidebar.selectbox(
        "Choose a model",
        options=list(MODELS.keys()),
        index=1  # Default to the finetuned model
    )
    # Store the selected model ID in session state
    st.session_state['current_model'] = MODELS[selected_model_name]
    
    # Padding between model selection and disclaimer
    st.sidebar.markdown("##")
    st.sidebar.caption(DISCLAIMER)
    st.sidebar.markdown("##")
    
    # Embed a button link to Google Form in the sidebar
    google_form_button = """
    <a href="https://forms.gle/jRpsrsL1TzRzaFgT6" target="_blank">
        <button style="color: white; background-color: #4CAF50; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 12px;">
            Provide Feedback
        </button>
    </a>
    """
    st.sidebar.markdown(google_form_button, unsafe_allow_html=True)

    # # Main page
    # # Header (centered)
    # with st.container():
    #     st.image(PANDAI_PNG)
    #     st.title("EdTech Model Tester")
        
def markdown_to_html_file(markdown_string, html_file_path):
    pypandoc.convert_text(markdown_string, 'html', format='md', outputfile=html_file_path)

def html_to_word(html_file_path, word_file_path):
    pypandoc.convert_file(html_file_path, 'docx', outputfile=word_file_path)

def create_download_link(word_file_path):
    with open(word_file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(word_file_path)}">Download Word File</a>'
    return href

def copy_to_clipboard():

    # Format the entire chat history with clear spacing
    chat_history = st.session_state.get('chat_history', [])
    entire_chat = "\n\n".join(f"{'Olier' if message['role'] == 'assistant' else 'User'}: {message['content']}" for message in chat_history if message['role'] in ['user', 'assistant'])

    with tempfile.TemporaryDirectory() as tmpdirname:
        html_file_path = os.path.join(tmpdirname, 'output.html')
        
        # Generate a neater timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H-%M")
        word_file_path = os.path.join(tmpdirname, f'EdTech{timestamp}.docx')
        
        # Convert the formatted chat history to HTML and then to a Word document
        markdown_to_html_file(entire_chat, html_file_path)
        html_to_word(html_file_path, word_file_path)
        
        # Create download link for the Word document
        b64 = base64.b64encode(open(word_file_path, 'rb').read()).decode()
        download_link = f'<a href="data:application/octet-stream;base64,{b64}" download="EdTechchat{timestamp}.docx" class="btn btn-primary" style="margin-top: 1em; margin-right: 0.5em; padding: 4px 12px; font-size: 14px; border-radius: 8px; cursor: pointer; border: 1px solid #f63366; display: inline-block; text-decoration: none; color: white; background-color: #f63366;">Download</a>'


        buttons_html = f"""
        <div style="text-align: right; margin-top: 1em;">
            <a href="data:application/octet-stream;base64,{b64}" download="EdTechchat{timestamp}.docx" style="text-decoration: none;">
                <button class="btn btn-primary" style="padding: 4px 12px; font-size: 14px; border-radius: 8px; cursor: pointer; border: 1px solid; display: inline-block;">
                    Download
                </button>
            </a>
        </div>
        """
        st.markdown(buttons_html, unsafe_allow_html=True)
# Add new function for Fireworks streaming
def stream_fireworks_response(messages, chat_history):
    try:
        # Create the assistant chat message container
        with st.chat_message("assistant", avatar=PANDAI_AVATAR_PNG):
            # Initialize the assistant's response placeholder
            assistant_response = ""
            assistant_message_placeholder = st.empty()
            
            # Fireworks LLM accepts messages in the same format
            # messages is already prepared
            
            # Stream the response
            for chunk in fireworks_llm.stream(messages):
                # chunk is an AIMessageChunk, extract the content
                delta = chunk.content
                if delta:  # Ensure that content is not None
                    assistant_response += delta
                    # Update the UI with the accumulated response
                    assistant_message_placeholder.markdown(assistant_response)
                
            # Once the response is complete, append it to the chat history
            if assistant_response:
                chat_history.append({"role": "assistant", "content": assistant_response})
                    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

# Update the stream_response function to handle both types
def stream_response(model, messages, chat_history):
    # Determine which streaming function to use based on the model type
    model_name = next(name for name, id in MODELS.items() if id == model)
    model_type = MODEL_TYPES[model_name]
    
    if model_type == "openai":
        try:
            with st.chat_message("assistant", avatar=PANDAI_AVATAR_PNG):
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.4,
                    stream=True
                )
                response = st.write_stream(stream)
                if response:
                    chat_history.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"An error occurred: {e}")
            print(f"An error occurred: {e}")
    else:  # fireworks
        stream_fireworks_response(messages, chat_history)
        
def truncate_chat_history(chat_history):
    # Always keep the system message
    truncated_history = [msg for msg in chat_history if msg["role"] == "system"]

    # Keep only the last 3 exchanges (6 messages) between User and Assistant
    user_assistant_messages = [msg for msg in chat_history if msg["role"] in ["user", "assistant"]]
    truncated_history.extend(user_assistant_messages[-6:])  # Keep last 3 exchanges

    return truncated_history

def run_chatbot():
    display_chat_ui()


    if 'chat_history' not in st.session_state:
        # Initialize chat history with a default system message
        st.session_state['chat_history'] = [{"role": "system", "content": "You are a helpful assistant."}]
        
    # Display existing chat history (excluding the system message)
    for chat in st.session_state['chat_history']:
        if chat["role"] != "system":  # Skip system messages
            with st.chat_message(
                chat["role"],
                avatar=PANDAI_AVATAR_PNG if chat["role"] == "assistant" else LOTUS_PNG,
            ):
                st.markdown(chat["content"])


    user_input = st.chat_input("Ask EdTech-Bot...")
    if user_input:
        # Add user's input to chat history and display it
        st.session_state['chat_history'].append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar=LOTUS_PNG):
            st.markdown(user_input)

        # Truncate the chat history before making a new API call
        st.session_state['chat_history'] = truncate_chat_history(st.session_state['chat_history'])

        # Prepare messages for API request
        messages = [{"role": chat['role'], "content": chat['content']} for chat in st.session_state['chat_history']]

        # Stream response from OpenAI
        stream_response(st.session_state['current_model'], messages, st.session_state['chat_history'])
        copy_to_clipboard()


    return st.session_state['chat_history']


run_chatbot()