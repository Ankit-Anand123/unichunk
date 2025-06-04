from PIL import Image, ImageOps
import io
import fitz
from ollama import chat
from state import IngestionState
from tqdm import tqdm

def visual_chunker(state: IngestionState) -> IngestionState:
    try:
        vlm_prompt = state.vlm_prompt
        model = state.vlm_model or "qwen2.5vl:7b"
        dpi = 200

        doc = fitz.open(state.file_path)
        visual_chunks = []

        for page_num, page in tqdm(enumerate(doc, start=1), total=len(doc), desc="Visual Chunking"):
            if not page.get_images():
                continue

            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("png")

            # Extract original size BEFORE resizing
            img = Image.open(io.BytesIO(img_bytes))
            original_width, original_height = img.size

            # Resize to 512x512 for VLM model
            target_size = (512, 512)
            img_resized = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            img_resized.save(buf, format="PNG")
            resized_img_bytes = buf.getvalue()

            response = chat(
                model=model,
                messages=[{
                    "role": "user",
                    "content": vlm_prompt,
                    "images": [resized_img_bytes]
                }]
            )
            vlm_output = response["message"]["content"]

            visual_chunks.append({
                "section_title": f"Visual Summary - Page {page_num}",
                "content": vlm_output,
                "page_numbers": [page_num],
                "coordinates": page.rect, 
                "normalization_factors": {
                    "original_width": original_width,
                    "original_height": original_height,
                    "resized_width": target_size[0],
                    "resized_height": target_size[1],
                },
                "chunk_id": f"visual_page_{page_num}",
                "chunk_type": "visual"
            })

        if state.image_chunks is None:
            state.image_chunks = []
        state.image_chunks.extend(visual_chunks)  
        return state

    except Exception as e:
        state.error = str(e)
        raise e
