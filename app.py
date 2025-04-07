import sys
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass
import os
import streamlit as st
import asyncio
import wolframalpha
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile

# Page configuration
st.set_page_config(page_title="Damen Technical Chatbot", layout="wide")

# Initialize session state for authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "attempts" not in st.session_state:
    st.session_state.attempts = 0

# Function to handle authentication
def authenticate():
    if st.session_state.password == st.secrets["APP_PASSWORD"]:
        st.session_state.authenticated = True
        st.session_state.attempts = 0
        return True
    else:
        st.session_state.attempts += 1
        if st.session_state.attempts >= 5:
            st.error("Maximum attempts reached. Please try again later.")
            st.stop()
        else:
            st.error(f"Incorrect password. {5 - st.session_state.attempts} attempts remaining.")
        return False

# Function to reset chat
def reset_chat():
    st.session_state.conversation = []
    st.session_state.vector_store = None
    st.success("Chat has been reset. You can upload a new document and start a new conversation.")

# Authentication screen
if not st.session_state.authenticated:
    st.title("Welcome to Damen Technical Chatbot")
    st.markdown("""
    This application provides access to technical documentation and equation analysis.
    Please enter the password to continue.
    """)
    
    st.text_input("Enter Password", type="password", key="password", on_change=authenticate)
    
    # Display login-related graphics or info
    st.markdown("---")
    st.markdown("### About Damen Technical Chatbot")
    st.markdown("""
    This chatbot is designed to analyze technical documents, including equations and parameters.
    It can assist with engineering calculations and provides access to technical specifications.
    """)
    st.stop()

# Main application (only runs if authenticated)
st.title("Damen Technical Chatbot")

# Setup environment variables from Streamlit secrets
@st.cache_resource
def setup_environment():
    os.environ['LANGSMITH_API_KEY'] = st.secrets["LANGSMITH_API"]
    os.environ['OPENAI_API_KEY'] = st.secrets["OPEN_AI_API"]
    os.environ['ANTHROPIC_API_KEY'] = st.secrets["CLAUDE_API"]
    
    wolfram_id = st.secrets["WOLFRAM_CLIENT"]
    client = wolframalpha.Client(wolfram_id)
    
    return client

client = setup_environment()

# Initialize models and embeddings
@st.cache_resource
def init_models():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    model_id = "claude-3-7-sonnet-20250219"
    
    llm1 = init_chat_model(model_id, model_provider="anthropic", temperature=0)
    llm2 = init_chat_model(model_id, model_provider="anthropic", temperature=0)
    llm3 = init_chat_model(model_id, model_provider="anthropic", temperature=0)
    
    return embeddings, llm1, llm2, llm3

embeddings, llm1, llm2, llm3 = init_models()

# Define the preprompt
preprompt_5 = """
# Core Rules
## No Assumptions Ever
- If a parameter/equation is missing, say:  
  "Parameter/equation '[X]' is not provided in the document. Please supply it."
- Never infer "typical" values, defaults, or unit conversions.
- If you make one assumption state it clearly

## Equation Handling
- For any equation in the document:  
  - Output the LaTeX formula only if all variables are confirmed 
  - If variables are missing, list them and stop. Example:  
    "Required variables: [X, Y, Z]. Missing: [Y, Z]."
- For equations not in the document:  
  "The document does not specify an equation for [quantity]. Please provide it."
- Use plain text for all non-equation content; reserve LaTeX exclusively for mathematical expressions.

## Input/Output Discipline
- Variables: Treat all inputs as case-sensitive (e.g., V ≠ v).  
- Units: Never convert units unless the document provides the conversion factor.  
- Calculations: Only compute after all inputs are confirmed.
- If from the conversation, you get a message saying "Sanity check" that contains a different value for an equation you calculated, use this new value

## Error States
- If the user asks for non-document analysis:  
  "This requires external data. Only document-derived analysis is permitted."
- If the request is ambiguous:  
  "Clarify whether this is a technical (document-based) or general question."

## Workflow Template
1. Extract the equation (if provided) or request it.
2. List all variables in the equation.
3. Check the document for each variable.
4. If any are missing, halt and request them.
5. If all are confirmed, substitute the values and output the equation in LaTeX code, wrapped appropriately, with surrounding text in plain format.
6. When performing calculations use three decimal places

**Failure to follow these rules will result in incorrect analysis. Be precise.**  
**Take your time to provide a correct answer**  
**Consider these instructions throughout the conversation**
"""

# Function to evaluate equations using Wolfram Alpha
async def evaluate_equation(latex_expr: str) -> str:
    """
    Evaluates a LaTeX expression using the Wolfram Alpha API and returns the result.

    :param latex_expr: A string containing the LaTeX expression to evaluate.
    :return: A string containing the result of the evaluation.
    """
    try:
        # Query Wolfram Alpha
        res = await client.aquery(latex_expr)
        for pod in res.pods:
            if pod.title in [
                'Result',
                'result',
                'Exact Result',
                'Exact result',
                'Decimal Approximation',
                'Decimal approximation',
                'Percentage',
                'Repeating decimal',
                'Repeating Decimal'
            ]:      
                for sub in pod.subpods:
                    return sub.plaintext
        return "No result found."
    except Exception as e:
        return f"An error occurred: {e}"

# Function to process uploaded JSON file and create vector store
def process_uploaded_file(uploaded_file):
    # Save uploaded file to a temporary file
    # with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
    #     temp_file.write(uploaded_file.getvalue())
    #     temp_path = temp_file.name
    
    # Create vector store
    db_name = "chroma_db"
    
    # Delete the database when adding new documents
    if os.path.exists(db_name):
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
    
    vector_store = Chroma(embedding_function=embeddings, persist_directory=db_name)
    
    # Load documents from JSON
    loader = JSONLoader(
        file_path=uploaded_file,
        jq_schema=".[].content",
        text_content=False,
    )
    
    docs = loader.load()
    
    # Define the size of chunks and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    
    # Index chunks
    _ = vector_store.add_documents(documents=all_splits)
    
    # Clean up temp file
    os.unlink(uploaded_file)
    
    return vector_store

# Streamlit UI components for sidebar
st.sidebar.title("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a JSON file", type=["json"])

# Add a reset button to the sidebar
if st.sidebar.button("Reset Chat", key="reset_button"):
    reset_chat()

# Initialize session state for storing conversation
if "conversation" not in st.session_state:
    st.session_state.conversation = []
    
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Process uploaded file
if uploaded_file and not st.session_state.vector_store:
    with st.spinner("Processing document..."):
        st.session_state.vector_store = process_uploaded_file(uploaded_file)
    st.success("Document processed successfully!")

# Display conversation
for message in st.session_state.conversation:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the uploaded document"):
    if not st.session_state.vector_store:
        st.error("Please upload a JSON file first.")
    else:
        # Add user message to conversation
        st.session_state.conversation.append({"role": "user", "content": preprompt_5})
        st.session_state.conversation.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Process user input and generate response
        with st.spinner("Thinking..."):
            # Retrieve relevant documents
            retrieved_docs = st.session_state.vector_store.similarity_search(prompt, k=5)
            docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
            
            # Get conversation history
            past_conversation = "\n".join([f"{'User' if msg['role'] == 'user' else 'LLM'}: {msg['content']}" 
                                          for msg in st.session_state.conversation[-16:]])
            
            # Generate response with LLM1
            full_prompt = f"""
            You are an assistant that remembers past messages. Use the conversation history below to stay consistent.

            Conversation so far:
            {past_conversation}

            Context:
            {docs_content}

            New Question:
            {prompt}
            """
            
            answer = llm1.invoke(full_prompt)
            response = answer.content
            
            # Process with LLM2 to identify equations
            second_prompt = f"""
            You are an assistant that analyzes equations in a provided text.
            Your job is:
            - Identify equations with fully replaced numerical values (no variables), only if they are already present in the provided text.
            - Do not make up your own values
            - Use notation that is friendly for wolfram alpha api for equations:
                For example use * for multiplication, sqrt(x), \\frac{{x}}{{y}}, etc.
            - Wrap each such equation in $$ symbols, one per line, no extra text.
            - If no such equations exist, return "No" as response.
            - If the equations are equivalent just include the first occurrence of it
                For example: y = 10 + 2 + 1 + 2, y = 12 + 1 + 2, only include the first definition of y
            - If an equation is written as follows: Y = X = 10 + 2, just include the first term Y = 10 + 2
            - Example of incorrect display:
              $$ y = 1 * 10 + 2 = 10 + 2 $$
              $$ y = 1 + 9 $$
              $$ x = 1 + 3 $$
            - Example of correct diplay
             $$ solve y = 1 * 10 + 2 $$
             $$ solve y = 1 + 9 $$
             $$ solve x = 1 + 3 $$
            - It is crutial you follow these instructions, failure to this would mean fatality
            Context:
            {response}
            """
            
            list_equations = llm2.invoke(second_prompt).content
            
            # Evaluate equations with Wolfram Alpha
            evaluated_results = []
            if "No" not in list_equations:
                for line in list_equations.split("\n"):
                    line = line.strip()
                    if "$" in line:
                        eq = line.replace("$$", "").strip()
                        if eq:
                            evaluated_eq = asyncio.run(evaluate_equation(eq))
                            evaluated_results.append(f"{eq} → {evaluated_eq}")
            
            # Combine results
            final_response = f"General Response:\n{response}\n"
            if evaluated_results:
                final_response += f"\nSanity check of correct calculations:\n" + "\n".join(evaluated_results)
            else:
                final_response += "No numerical equations found to evaluate."
            
            # Process with LLM3 for final refinement
            third_prompt = f"""
            You are an assistant with the only task of refining a previous response, you will have a structure as follows:
            Your job is crutial to the whole framework so please make sure you stick to what is asked
            "General Response
            ...."
            Then either:
            "Sanity check" or "No numerical equations found to evaluate"
            Your job is:
            - Identify equations that are equivalent from the second section of the response if there was a sanity check
            - Only retrieve the first equation with its value, as it may be more than one equation for the same parameter
            - Remove all the other equivalent equations if any
            - Then using the new value from the sanity check, you job is to find that same value in the "General response" and replace it there
            - Do not change notation
            - If no such equations exist, just return the original response you got without any change
            - It is crutial you follow these instructions, failure to this would mean fatality
            Context:
            {final_response}
            """
            
            final_output = llm3.invoke(third_prompt).content
            
            # Add assistant message to conversation
            st.session_state.conversation.append({"role": "assistant", "content": final_output})
            st.chat_message("assistant").write(final_output)

# Add information about the app
st.sidebar.markdown("---")
st.sidebar.info(
    """
    This chatbot analyzes technical documents containing equations and parameters.
    Upload a JSON file in the required format to begin.
    
    The chatbot can:
    - Answer questions about technical documents
    - Process equations and verify calculations
    - Provide precise technical information
    
    Use the "Reset Chat" button to clear the conversation and start a new session.
    """
)

# Add a logout option
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.attempts = 0
    st.session_state.conversation = []
    st.session_state.vector_store = None
    st.rerun()