# DamenChat

A Streamlit-based chatbot application for analyzing technical documents, equations, and parameters. This application uses RAG (Retrieval-Augmented Generation) to provide accurate responses based on uploaded JSON documents.

## Features

- **Document Analysis**: Upload JSON files containing technical documentation and equations
- **Equation Processing**: Automatically identify and evaluate equations in the documents
- **Wolfram Alpha Integration**: Verify calculations using Wolfram Alpha
- **Conversation Memory**: Maintains context throughout the conversation
- **Password Protection**: Secure access to the application

## Setup Instructions

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your API keys in the `.streamlit/secrets.toml` file:
   ```
   "LANGSMITH_API" = "your_langsmith_api_key"
   "OPEN_AI_API" = "your_openai_api_key"
   "CLAUDE_API" = "your_anthropic_api_key"
   "WOLFRAM_CLIENT" = "your_wolfram_client_id"
   "APP_PASSWORD" = "your_app_password"
   ```

## Running the Application

To run the application, use the following command:

```
streamlit run streamlit_chatbot.py
```

## Usage

1. Enter the password to access the application
2. Upload a JSON file containing technical documentation
3. Ask questions about the uploaded document
4. The chatbot will analyze the document, identify equations, and provide accurate responses
5. Use the "Reset Chat" button to start a new conversation
6. Use the "Logout" button to exit the application

## Troubleshooting

- **Truncated Responses**: If you notice that responses are being cut off, try asking more specific questions or breaking down complex queries into smaller parts.
- **Memory Issues**: The application maintains conversation history, but if you experience memory-related issues, use the "Reset Chat" button to start fresh.
- **Equation Evaluation**: If equations are not being evaluated correctly, ensure they are formatted properly in the JSON file.

## Technical Details

- Uses Claude-3-7-Sonnet for natural language processing
- Implements RAG with Chroma vector database
- Integrates with Wolfram Alpha for equation evaluation
- Uses Streamlit for the web interface

## Future Improvements

- Support for additional document formats
- Enhanced equation visualization
- Improved conversation memory management
- Additional security features

TODO:
- Create a new environment, try the requirements txt file and see if it runs with the current libraries -> DONE, requirements.txt is now ready.
- The memory is not working for some reason, the first answer works, but the second one is not
- The app.py has the same problem as streamlit_chatbot.py which is my testing file.
- Modify the code so that we can read a new document using mathpix 


pip install -r requirements.txt