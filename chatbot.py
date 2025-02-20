import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

load_dotenv()

class PDFChatbot:
    def __init__(self):
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Initialize Perplexity AI model
        self.llm = ChatOpenAI(
            model="sonar",
            temperature=0.7,
            max_tokens=1000
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize vectorstore and document-related attributes
        self.vectorstore = None
        self.documents = []
        self.page_content = []

    def process_pdf(self, pdf_file):
        try:
            # Save uploaded file temporarily
            with open("temp.pdf", "wb") as f:
                f.write(pdf_file.getbuffer())

            # Load and process the PDF
            loader = PyPDFLoader("temp.pdf")
            self.documents = loader.load()

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(self.documents)
            # Store page contents for reference
            self.page_content = [doc.page_content for doc in splits]

            # Create vectorstore
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)

            # Remove temporary file
            os.remove("temp.pdf")
            return "PDF processed successfully!"
        
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return f"Error: {str(e)}"

    def retrieve_context(self, query:str, top_k=3):
        """Retrieve most relevant document chunks"""
        if not self.vectorstore:
            return ""
        
        # Perform similarity search
        relevant_docs = self.vectorstore.similarity_search(query, k=top_k)
        
        # Combine retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        return context

    def get_summary(self):
        if not self.vectorstore:
            return "Please upload a PDF first!"

        try:
            # Create a prompt for summarization
            summary_prompt = PromptTemplate.from_template(
                """You are an expert document analyzer. 
                Based on the following context, provide a comprehensive summary:

                Context:
                {context}

                Summary:"""
            )
            # Create LLM chain for summarization
            summary_chain = summary_prompt | self.llm
            # Get context
            context = self.retrieve_context(query="summarize this document", top_k=3)

            # Generate summary
            summary = summary_chain.invoke({"context":context})
            return summary.content
        
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")
            return f"Error: {str(e)}"

    def ask_question(self, question):
        if not self.vectorstore:
            return "Please upload a PDF first!"

        try:
            # Create a prompt for question answering
            qa_prompt = PromptTemplate(
                input_variables=["chat_history", "context", "question"],
                template="""You are an AI research assistant helping to explain content from document in an easy and conciseway.

                Chat History:
                {chat_history}

                Context:
                {context}

                Question: {question}
                Helpful Answer:"""
            )

            # Retrieve relevant context
            context = self.retrieve_context(question)

            # Create LLM chain for Q&A
            qa_chain = LLMChain(
                llm=self.llm, 
                prompt=qa_prompt
            )

            # Get chat history
            history = self.memory.load_memory_variables({})['chat_history']
            chat_history = "\n".join([str(msg) for msg in history])

            # Generate answer
            answer = qa_chain.run(
                chat_history=chat_history,
                context=context,
                question=question
            )

            # Update memory
            self.memory.save_context(
                {"input": question},
                {"output": answer}
            )

            return answer
        
        except Exception as e:
            st.error(f"Error answering question: {str(e)}")
            return f"Error: {str(e)}"

# Streamlit UI
def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📜")
    
    st.title("📜 PDF Summarizing Chatbot")
    st.write("Upload a PDF and explore its contents!")
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = PDFChatbot()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    # PDF Upload Section
    # col1, col2 = st.columns([8,4])
    # with col1:
    st.markdown("""
        <style>
        .stFileUploader {
            width: 90%;  # Adjust width as needed
        }
        </style>
    """, unsafe_allow_html=True)
    pdf_file = st.file_uploader("Upload PDF", type=['pdf'])
    
    if pdf_file:
        # with col2:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                if not st.session_state.pdf_processed:
                    result = st.session_state.chatbot.process_pdf(pdf_file)
                    
                    if "successfully" in result:
                        st.session_state.pdf_processed = True
                    
                # Generate and display summary
                with st.spinner("Generating summary..."):
                    if not st.session_state.summary:
                        summary = st.session_state.chatbot.get_summary()
                        st.session_state.summary = summary
                    st.success("Summary Generated!")
                        
                    # Expandable summary
                    st.markdown("""
                        <style>
                        .stExpander {
                            width: 100%;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    with st.expander("View Summary"):
                        st.write(st.session_state.summary)

    # Chat Interface
    st.header("Chat with PDF")
    
    # Question input
    question = st.text_input("Ask a question about the PDF:", 
                              disabled=(not st.session_state.pdf_processed))
    
    # Ask button
    if st.button("Ask", disabled=(not st.session_state.pdf_processed)):
        if question:
            with st.spinner("Generating response..."):
                answer = st.session_state.chatbot.ask_question(question)
                
                # Update chat history
                st.session_state.chat_history.append(("You", question))
                st.session_state.chat_history.append(("Bot", answer))

                # Display response
                st.markdown("## Response")
                st.markdown(f"**Bot** 🤖: {answer}")

    # Chat History
    st.header("Chat History")
    val=-2 if len(st.session_state.chat_history)>2 else 0
    for role, text in (st.session_state.chat_history[:val]):
        if role == "You":
            st.markdown(f"**You** 😁: {text}")
        else:
            st.markdown(f"**Bot** 🤖: {text}")

    # Utility Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("Reset PDF"):
            st.session_state.pdf_processed = False
            st.session_state.chatbot = PDFChatbot()
            st.rerun()

if __name__ == "__main__":
    main()