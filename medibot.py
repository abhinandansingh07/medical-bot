import os
import streamlit as st

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from huggingface_hub import InferenceClient

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""


class HuggingFaceAPIEmbeddings(Embeddings):
    def __init__(self, api_key, model_name):
        self.client = InferenceClient(provider="hf-inference", api_key=api_key)
        self.model_name = model_name

    def embed_documents(self, texts):
        embeddings = self.client.feature_extraction(texts, model=self.model_name)
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.client.feature_extraction(text, model=self.model_name)
        return embedding.tolist()


@st.cache_resource
def get_vectorstore():
    try:
        hf_token = os.getenv("HF_TOKEN")

        if not hf_token:
            st.error("HF_TOKEN was not found. Add it to your Render environment variables or local .env file.")
            st.stop()

        embedding_model = HuggingFaceAPIEmbeddings(
            api_key=hf_token,
            model_name='sentence-transformers/all-MiniLM-L6-v2',
        )
        db = FAISS.load_local(
            DB_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        return db
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None


def set_custom_prompt(template):
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


def get_qa_chain(vectorstore):
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        st.error("GROQ_API_KEY was not found. Please add it to your .env file or Render environment variables.")
        st.stop()

    llm = ChatGroq(
        model="llama-3.1-8b-instant",   # current free + fast Groq model
        temperature=0.0,
        groq_api_key=groq_api_key,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={
            'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
        }
    )

    return qa_chain


def format_source_docs(source_documents):
    sources_text = "\n\n---\n📄 **Sources:**\n"
    for i, doc in enumerate(source_documents, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        sources_text += f"\n**{i}.** `{source}` — Page {page}"
    return sources_text


def main():
    st.title("🏥 MediBot - Ask Me Anything!")
    st.caption("Medical information chatbot powered by Groq + LLaMA3")

    # Initialize session state.
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Load the cached vector store.
    vectorstore = get_vectorstore()
    if vectorstore is None:
        st.error("The vector store could not be loaded. Please rebuild the vector store first.")
        st.stop()

    # Build the QA chain.
    try:
        qa_chain = get_qa_chain(vectorstore)
    except Exception as e:
        st.error(f"Error building QA chain: {str(e)}")
        st.stop()

    # Display previous chat messages.
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Accept user input.
    prompt = st.chat_input("Type your question here...")

    if prompt:
        # Display the user message.
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Generate the response.
        with st.spinner("Thinking..."):
            try:
                response = qa_chain.invoke({'query': prompt})

                result = response.get("result", "No answer was found.")
                source_documents = response.get("source_documents", [])

                # Add sources if available.
                if source_documents:
                    result_to_show = result + format_source_docs(source_documents)
                else:
                    result_to_show = result

                st.chat_message('assistant').markdown(result_to_show)
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': result_to_show
                })

            except Exception as e:
                error_msg = str(e)
                st.error(f"Error: {error_msg}")

                # Helpful messages for common errors.
                if "fields_set" in error_msg:
                    st.info("💡 Fix: run `pip install pydantic==2.7.1 langchain-groq==0.1.9`")
                elif "expecting value" in error_msg.lower():
                    st.info("💡 Fix: check HF_TOKEN. Add your Hugging Face token to Render or your local .env file.")
                elif "api_key" in error_msg.lower():
                    st.info("💡 Fix: check GROQ_API_KEY in your .env file or Render environment variables.")
                elif "faiss" in error_msg.lower():
                    st.info("💡 Fix: run `pip install faiss-cpu`")


if __name__ == "__main__":
    main()
