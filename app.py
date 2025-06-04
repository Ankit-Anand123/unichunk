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
st.set_page_config(page_title="UNICHUNK - PDF Insight Assistant", layout="wide")
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
        
        # Debug info
        st.write(f"Found {len(result['highlights'])} highlight(s)")
        
        if st.session_state.get("pdf_bytes"):
            try:
                doc = fitz.open(stream=st.session_state["pdf_bytes"], filetype="pdf")
                
                # Process each highlight individually
                for i, highlight_item in enumerate(result["highlights"]):
                    page_nums = highlight_item.get("page_numbers", [])
                    coords = highlight_item.get("coordinates")
                    chunk_id = highlight_item.get("chunk_id", f"highlight_{i}")
                    element_category = highlight_item.get("element_category", "unknown")
                    distance = highlight_item.get("distance", "N/A")
                    
                    # Debug info
                    st.write(f"**Highlight {i+1}:** {element_category} on page {page_nums} (distance: {distance:.3f})")
                    
                    if not page_nums or not coords:
                        st.warning(f"Skipping highlight {i+1}: missing page or coordinates")
                        continue
                    
                    page_num = page_nums[0]  # Should be exactly one page now
                    
                    if page_num < 1 or page_num > len(doc):
                        st.warning(f"Page {page_num} out of range")
                        continue
                    
                    try:
                        page = doc[page_num - 1]  # Zero-indexed
                        page_rect = page.rect
                        
                        # Parse coordinates for this specific element
                        from tools import parse_coordinates, validate_and_clamp_rectangle, debug_coordinates
                        
                        # Debug the coordinate structure
                        debug_coordinates(coords, chunk_id)
                        
                        rects = parse_coordinates(coords)
                        st.write(f"  Parsed {len(rects)} rectangle(s)")
                        
                        valid_rects = []
                        for j, rect in enumerate(rects):
                            validated_rect = validate_and_clamp_rectangle(rect, page_rect)
                            if validated_rect:
                                valid_rects.append(validated_rect)
                                st.write(f"    Rect {j+1}: ({rect.x0:.1f}, {rect.y0:.1f}, {rect.x1:.1f}, {rect.y1:.1f}) -> Valid")
                            else:
                                st.write(f"    Rect {j+1}: ({rect.x0:.1f}, {rect.y0:.1f}, {rect.x1:.1f}, {rect.y1:.1f}) -> Invalid")
                        
                        if valid_rects:
                            # Add highlights to page
                            for rect in valid_rects:
                                highlight_annot = page.add_highlight_annot(rect)
                                highlight_annot.set_colors(stroke=[1, 1, 0])  # Yellow highlight
                                highlight_annot.update()
                            
                            # Render page with highlights at high resolution
                            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom
                            pix = page.get_pixmap(matrix=mat)
                            img_bytes = pix.tobytes("png")
                            img = Image.open(io.BytesIO(img_bytes))
                            
                            # Save and display
                            output_path = f"highlighted_pages/page_{page_num}_highlight_{i}.png"
                            img.save(output_path)
                            
                            st.image(img, 
                                    caption=f"Page {page_num} - {element_category} - {chunk_id}", 
                                    width=800)
                            
                            st.success(f"✅ Successfully highlighted {len(valid_rects)} area(s) on page {page_num}")
                        else:
                            st.error(f"❌ No valid rectangles found for page {page_num}")
                            
                    except Exception as e:
                        st.error(f"Error processing page {page_num}: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                
                doc.close()
                
            except Exception as e:
                st.error(f"Error processing PDF for highlighting: {e}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning("PDF not available for highlighting")