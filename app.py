import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
import openai
import os
import logging

# Set up logging and load environment variables
logging.basicConfig(level=logging.DEBUG)
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize API wrappers and tools for Wikipedia and Arxiv
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
search = DuckDuckGoSearchRun(name="Search")

# Streamlit UI setup
st.title("🔎 LangChain - Chat with Search")
st.write("""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
""")

# Sidebar settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "Hi, I'm a chatbot who can search the web. How can I help you?"
    }]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Process user input
if prompt := st.chat_input(placeholder="What is Generative AI?"):
    # Append and display the user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Initialize the LLM and agent tools
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki]
    
    # Create the agent with increased iteration and time limits, and enable parsing error handling
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        max_iterations=100,
        max_execution_time=30,
        verbose=True
    )
    
    # Use the latest user message as input
    user_input = st.session_state.messages[-1]["content"]
    
    # Set up the Streamlit callback handler
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    
    # Retry mechanism for handling parsing or iteration issues
    max_retries = 3
    retry_count = 0
    response = ""
    
    while retry_count < max_retries and not response:
        try:
            response = search_agent.run(user_input, callbacks=[st_cb])
        except ValueError as e:
            logging.error(f"Error: {e}")
            retry_count += 1
            response = "I couldn't find relevant information. Try refining your question."
    
    # Append and display the assistant's response
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
