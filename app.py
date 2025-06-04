import streamlit as st
from streamlit_backend_connector import run_ingestion, run_query
from tools import parse_coordinates
import os
import tempfile
import base64
import pandas as pd
from PIL import Image
import io
import fitz
import time
import os

os.makedirs("highlighted_pages", exist_ok=True)

# Page setup
st.set_page_config(page_title="PDF Insight Assistant", layout="wide")
st.title("\U0001F4D8 PDF Insight Assistant")

# -------------------- Sidebar Session Control --------------------
with st.sidebar:
    st.header("\U0001F4CA Session Info")
    if st.session_state.get("pdf_processed"):
        st.success("✅ PDF Processed")
        st.metric("Total Q&A Pairs", len(st.session_state.get("conversation_history", [])))
        if st.button("🗑️ Reset Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    else:
        st.info("📤 Please upload a PDF to get started")

# Initialize session state if not present
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

if "input_key" not in st.session_state:
    st.session_state["input_key"] = 0

# -------------------- PDF Upload Section --------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and not st.session_state.get("pdf_processed"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    # Progress bar with detailed steps
    with st.spinner("Starting ingestion pipeline..."):
        progress = st.progress(0, text="Initializing...")
        time.sleep(0.5)

        progress.progress(10, text="Uploading file...")
        time.sleep(0.5)

        progress.progress(30, text="Chunking text and images...")
        ingestion_result = run_ingestion(file_path=temp_pdf_path)
        progress.progress(60, text="Generating embeddings...")
        time.sleep(0.5)

        progress.progress(80, text="Storing into vector database...")
        time.sleep(0.5)

        progress.progress(100, text="Ingestion complete!")
        time.sleep(0.5)

    os.remove(temp_pdf_path)

    st.session_state["uploaded_pdf_path"] = temp_pdf_path
    st.session_state["pdf_processed"] = True
    st.success("✅ PDF successfully processed and indexed!")

# -------------------- Conversation History --------------------
if st.session_state["conversation_history"]:
    st.subheader("\U0001F4AC Conversation History")
    for i, qa_pair in enumerate(st.session_state["conversation_history"]):
        with st.expander(f"Q{i+1}: {qa_pair['question'][:50]}{'...' if len(qa_pair['question']) > 50 else ''}", expanded=(i == len(st.session_state["conversation_history"])-1)):
            st.markdown(f"**❓ Question:** {qa_pair['question']}")
            st.markdown(f"**📢 Answer:** {qa_pair['answer']}")
            st.markdown("---")

# -------------------- Question Input --------------------
if st.session_state.get("pdf_processed"):
    st.subheader("❓ Ask a Question")

    col1, col2 = st.columns([4, 1])
    with col1:
        user_query = st.text_input("Type your question about the document here...", key=f"user_input_{st.session_state['input_key']}")

    with col2:
        ask_button = st.button("Ask", type="primary")
        clear_button = st.button("Clear History")

    if clear_button:
        st.session_state["conversation_history"] = []
        st.rerun()

    if (ask_button or user_query) and user_query.strip():
        st.info("Searching and generating answer...")

        query_result = run_query(user_query)
        st.session_state["conversation_history"].append({
            'question': user_query,
            'answer': query_result["answer"],
            'result': query_result
        })
        st.session_state["input_key"] += 1
        st.rerun()

# -------------------- Display Last Answer --------------------
if st.session_state["conversation_history"]:
    last_record = st.session_state["conversation_history"][-1]
    st.subheader("📢 Latest Answer")
    st.write(last_record["answer"])

    result = last_record["result"]

    # Show dataframe if available
    if result.get("dataframe") is not None and not result["dataframe"].empty:
        st.subheader("📊 Retrieved Table")
        st.dataframe(result["dataframe"])

    # Show plot if available
    if result.get("plot_data"):
        st.subheader("📊 Generated Plot")
        for encoded_img in result["plot_data"]:
            img_bytes = base64.b64decode(encoded_img)
            img = Image.open(io.BytesIO(img_bytes))
            st.image(img)

    # Show highlight info if available
    if result.get("highlights"):
        st.subheader("📍 Highlighted Pages")
        if uploaded_file:
            pdf_file = uploaded_file
        elif st.session_state.get("uploaded_pdf_path"):
            pdf_file = open(st.session_state["uploaded_pdf_path"], "rb")
        else:
            pdf_file = None

        if pdf_file:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            for item in result["highlights"]:
                page_num = item["page_numbers"][0]
                coords = item["coordinates"]
                rects = parse_coordinates(coords)
                
                page = doc[page_num - 1]
                for rect in rects:
                    page_rect = page.rect
                    safe_rect = rect & page_rect 
                    # Only attempt extraction if safe_rect has non-zero area
                    if safe_rect.is_empty or safe_rect.get_area() == 0:
                        print(f"Skipping rectangle completely outside page: {safe_rect}")
                        continue
                    page.add_highlight_annot(safe_rect)
                    pix = page.get_pixmap()
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    output_path = f"highlighted_pages/page_{page_num}.png"
                    img.save(output_path)
                    st.image(img, caption=f"Page {page_num} with Highlight")
            pdf_file.close()
