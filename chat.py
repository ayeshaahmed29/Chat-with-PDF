from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

PDF_PATH= r"C:\Users\PMYLS\Downloads\attention-is-all-you-need.pdf"
CHROMA_DIRECTORY="./chroma_db"

#loading PDF
print("loading PDF...")
loader= PyPDFLoader(PDF_PATH)
document= loader.load()
print(f"loaded {len(document)} pages")

#splitting document
print("splitting document into chunks...")
splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks= splitter.split_documents(document)
print(f"split into {len(chunks)} chunks")

#embedding 
print("innitializing Gemini's embedding model")
embeddings= GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#creating and populating Chroma DB
if os.path.exists(CHROMA_DIRECTORY) and os.listdir(CHROMA_DIRECTORY):
    print("Loading existing Chroma DB...")
    vectorstore = Chroma(persist_directory=CHROMA_DIRECTORY, embedding_function=embeddings)
    print("Chroma DB loaded.")
else:
    print("Creating new Chroma DB and embedding chunks ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIRECTORY
    )
    print("Chroma DB created and persisted.")

#RAG implementation

#Initialize Gemini LLM 
LLM= ChatGoogleGenerativeAI(model= "gemini-1.5-flash", temperature=0.2)
print("using LLM: gemini-pro")

#creating retriever
retriever= vectorstore.as_retriever(search_kwargs={"k":3})

#building RAG chain
prompt = ChatPromptTemplate.from_template("""You are an AI assistant tasked with answering questions about a PDF document.
Answer the following question based ONLY on the provided context.
If the answer cannot be found in the context, respond with "I cannot find the answer to that question in the provided document."
Do not make up information.

<context>
{context}
</context>

Question: {input}
""")

document_chain = create_stuff_documents_chain(LLM, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)
print("RAG chain assembled.")

#for user interaction and testing 
print("\n---Chat with your PDF---")
print("type 'bye' to quit")

if __name__=="__main__":
    while True:
        question= input("\n Your Question: ")
        if question.lower()=='bye':
            break
        
        try:
            response= rag_chain.invoke({"input": question})
            print("\n Answer: ", response["answer"])
        except Exception as e:
            print(f"An error occurred: {e}")

    print("Exiting chat...Goodbye!")