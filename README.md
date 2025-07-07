<h1>Chat with Your PDF (Gemini & ChromaDB)</h1>
This project implements a Retrieval Augmented Generation (RAG) system that allows you to chat with the content of a PDF document. You can upload a research paper, textbook, or any PDF, and then ask questions about its content. The system will retrieve relevant information from the PDF and use a Large Language Model (LLM) to generate coherent and accurate answers.
<h2>Features</h2>
<ul>
  <li><b>PDF Content Ingestion:</b> Loads and processes PDF documents.</li>
  <li><b>Intelligent Text Chunking:</b> Splits documents into manageable chunks for efficient retrieval.</li>
  <li><b>Vector Database Storage:</b> Stores text embeddings in ChromaDB for fast semantic search.</li>
  <li><b>Google Gemini LLM Integration:</b> Utilizes Google's powerful Gemini models for natural language understanding and generation.</li>
  <li><b>Contextual Question Answering:</b> Answers questions based only on the content of the provided PDF, reducing irrelevant or fabricated responses.</li>
  <li><b>Persistent Storage:</b> ChromaDB allows the vector store to be saved to disk, so you don't have to re-embed the PDF every time you run the application.</li>
</ul>
<h2>Technologies Used</h2>
<ul>
<li><b>Python</b>: The core programming language.</li>
<li><b>LangChain</b>: Framework for building LLM applications.</li>
<li><b>langchain-google-genai & google-generativeai</b>: For integrating with Google Gemini models (LLM and Embeddings).</li>
<li><b>chromadb</b>: The vector database used for storing and retrieving document embeddings.</li>
<li><b>pypdf</b>: For loading and extracting text from PDF documents.</li>
<li><b>RecursiveCharacterTextSplitter</b>: For splitting documents into optimal chunks.</li>
</ul>
