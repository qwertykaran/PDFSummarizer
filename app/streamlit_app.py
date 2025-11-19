import streamlit as st
import ollama
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Your prompt template with reasoning
PROMPT_TEMPLATE = """
You are given a set of document excerpts.

Context:
{context}

Question:
{question}

Instructions:
- Answer ONLY from the provided context.
- If the answer is explicitly present, return it exactly as written.
- Do not infer, rephrase, or guess.
- If the context does not contain the answer, respond with: "Not found in context."

Answer:
"""

# Streamlit UI
st.title("📄 Local LLM Document Q&A with Ollama")

question = st.text_input("Enter your question:")
uploaded_file = st.file_uploader("Upload a document (.txt)", type=["txt"])

if uploaded_file and question:
    # Load document text
    doc_text = uploaded_file.read().decode("utf-8")

    # Split into chunks (basic splitter)
    chunks = doc_text.split("\n\n")
    relevant_chunks = chunks[:5]  # later replace with embeddings search

    # Concatenate context
    context_text = "\n\n---\n\n".join(relevant_chunks)

    # Create prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=question)

    # Call Ollama
    with st.spinner("Thinking..."):
        response = ollama.chat(
            model="mistral",  # change to llama2, qwen, etc.
            messages=[{"role": "user", "content": prompt}],
        )

    # Display result
    answer = response["message"]["content"]
    st.subheader("Answer")
    st.write(answer)

    # Reasoning part (simple extraction)
    st.subheader("Reasoning")
    if "Not found in context" in answer:
        st.write("The answer was not explicitly found in the provided context.")
    else:
        st.write("The answer was extracted directly from the document context provided.")
