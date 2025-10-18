# build-llm-apps

## Overview
This repository is designed to help you explore the LangChain ecosystem and build powerful applications using Large Language Models (LLMs). It includes exercises, utilities, and pipelines for Retrieval-Augmented Generation (RAG).

## Features
- **Chatbot**: Implementations for conversational AI.
- **RAG Pipeline**: Scripts for loading, splitting, embedding, and storing documents.
- **Exercises**: Hands-on examples to understand LangChain components.

## Setup

### Prerequisites
- Python 3.8 or higher
- Pipenv for managing dependencies

### Installation
1. Clone the repository:
   ```powershell
   git clone https://github.com/durga61/langchain-course.git
   ```
2. Navigate to the project directory:
   ```powershell
   cd langchain-course
   ```
3. Install dependencies using Pipenv:
   ```powershell
   pipenv install
   ```
4. Activate the virtual environment:
   ```powershell
   pipenv shell
   ```

## Usage

### Running Examples
Navigate to the `src/exercises` folder and run the Python scripts to explore different LangChain functionalities. For example:
```powershell
python prompt_template_llm_invoke.py
```

### RAG Pipeline
The `src/rag_pipeline` folder contains scripts for building a Retrieval-Augmented Generation pipeline. Start with `load_pdf_document.py` to load documents and proceed with `rag_chain.py` for querying.

## Contributing
Feel free to open issues or submit pull requests to improve this repository.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
