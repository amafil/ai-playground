# helpdesk-rag-playground

## Overview

The `helpdesk-rag-playground` is a Python-based project designed to provide a helpdesk solution using Retrieval-Augmented Generation (RAG). It integrates OpenAI's APIs for natural language processing and FAISS for efficient similarity-based retrieval. The system processes user queries, classifies them, and provides answers based on a predefined FAQ knowledge base.

---

## Features

- **FAQ Embedding and Retrieval**: Uses FAISS to index and retrieve FAQ answers based on similarity to user queries.
- **Message Classification**: Classifies incoming messages into predefined categories (e.g., assistance, error, information).
- **Keyword Extraction**: Extracts relevant keywords from user messages using YAKE or OpenAI's models.
- **Chat Completion**: Generates responses to user queries using OpenAI's chat completion API.
- **Helpdesk Message Handling**: Processes messages retrieved from a helpdesk channel.

---

## Project Structure

### 1. `main.py`

The main entry point of the application. It orchestrates the following:
- Initializes the OpenAI client and FAISS index.
- Retrieves helpdesk messages.
- Classifies messages and extracts keywords.
- Searches for answers in the FAQ knowledge base.
- Generates responses using OpenAI's chat completion API.

### 2. `core/`

This directory contains core functionalities of the project:
- **`embeddings.py`**: Handles embedding generation and FAISS index initialization.
- **`knowledge.py`**: Contains the FAQ knowledge base as a list of question-answer tuples.
- **`classification.py`**: Provides message classification and keyword extraction functionalities.
- **`utils.py`**: Includes utility functions, such as a mock function to retrieve helpdesk messages.

### 3. `services/`

This directory contains service-level functionalities:
- **`chat.py`**: Implements chat completion using OpenAI's API.
- **`search.py`**: Implements vectorial search for FAQ answers using FAISS.

### 4. `requirements.txt`

Lists the dependencies required for the project:
- `openai`: For interacting with OpenAI's APIs.
- `numpy`: For numerical computations.
- `faiss-cpu`: For similarity-based retrieval.
- `yake`: For keyword extraction.

### 5. `LICENSE`

The project is licensed under the GNU General Public License v3.0.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/helpdesk-rag-playground.git
   cd helpdesk-rag-playground
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Set up the required environment variables:
   - `OPENAI_BASE_URL`: Base URL for the OpenAI API.
   - `OPENAI_API_KEY`: API key for OpenAI.

2. Run the main script:
   ```bash
   python main.py
   ```

3. The script will:
   - Retrieve helpdesk messages.
   - Classify and process each message.
   - Search for relevant answers in the FAQ knowledge base.
   - Generate and print responses.

---

## FAQ Knowledge Base

The FAQ knowledge base is defined in `core/knowledge.py` as a list of tuples. Each tuple contains:
- A question (string).
- A corresponding answer (string).

Example:
```python
knowledge = [
    (
        "Where can I download the app?",
        "You can download it from the official stores..."
    ),
    ...
]
```

---

## Key Components

### FAISS Index

- Used for efficient similarity-based retrieval of FAQ answers.
- Initialized in `core/embeddings.py` with a dimension matching the embedding model output.

### OpenAI Integration

- **Embedding generation**: Converts text into vector representations.
- **Chat completion**: Generates conversational responses based on the FAQ knowledge base.

### Message Classification

- Categorizes messages into predefined types (e.g., assistance, error, information) using OpenAI's chat completion API.

### Keyword Extraction

- Extracts relevant keywords from messages using YAKE or OpenAI's API.

---

## Debugging

- Debug logs are printed to the console for:
  - Message classification.
  - Keyword extraction.
  - Chat completion requests and responses.

---

## Contributing

1. Fork the repository.
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push them to your fork.
4. Submit a pull request.

---

## License

This project is licensed under the GNU General Public License v3.0. See the `LICENSE` file for details.
