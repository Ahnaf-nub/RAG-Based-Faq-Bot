# FAQ Bot with Retrieval Augmented Generation (RAG)

This project implements a FAQ bot using Retrieval Augmented Generation (RAG) with OpenAI's GPT-4 and Pinecone as the vector database. The bot retrieves relevant information from a knowledge base (FAQs stored in Pinecone) and generates contextually accurate responses using GPT-4.

## Features

- **Retrieval Augmented Generation (RAG)**: Combines retrieval of relevant information with language generation.
- **FastAPI**: Provides a web interface for interacting with the bot.
- **Pinecone**: Stores and retrieves FAQ data using vector embeddings.
- **OpenAI GPT-4**: Generates responses based on retrieved information.
- **Static Web Interface**: Allows users to interact with the bot through a web page.

## Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key

4. **Create a `.env` file** with your API keys:

    ```plaintext
    OPENAI_API_KEY="your_openai_api_key"
    PINECONE_API_KEY="your_pinecone_api_key"
    ```

## Usage

1. **Run the code**

2. **Access the web interface**:

    Open your browser and go to [http://localhost:8000](http://localhost:8000).

3. **Interact with the bot**:

    Type your questions in the input box and click "Send" to get responses from the bot.

## Project Structure

- `main.py`: The main script that sets up the FastAPI server, initializes Pinecone, and configures the RAG pipeline.
- `faqs.txt`: The text file containing the FAQ data.
- `static/`: Directory containing static files for the web interface.
- `requirements.txt`: List of dependencies required for the project.
- `.env`: Environment variables file containing API keys.