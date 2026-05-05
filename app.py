"""
Streamlit Web Application for AttriDiffuser
Text-to-Face Generation Interface
"""
import streamlit as st
import torch
from PIL import Image
import os
from model import SimplifiedAttriDiffuser
from utils import extract_dataset, parse_attributes
import time
from datetime import datetime

# ── CLIP Score helper ──────────────────────────────────────────────────────────
@st.cache_resource
def load_clip():
    """Load CLIP model once and cache it."""
    from transformers import CLIPProcessor, CLIPModel
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    return clip_model, clip_processor

def compute_clip_score(image: Image.Image, text: str) -> float:
    """Return cosine similarity between image and text embeddings (0-1)."""
    try:
        clip_model, clip_processor = load_clip()
        inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = clip_model(**inputs)
        score = outputs.logits_per_image[0][0].item()
        # Normalise to 0-1 range (raw logit is typically 0-100)
        return round(min(max(score / 100.0, 0.0), 1.0), 3)
    except Exception as e:
        print(f"CLIP score error: {e}")
        return None
# ──────────────────────────────────────────────────────────────────────────────

# Page configuration
st.set_page_config(
    page_title="AttriDiffuser - Face Generation",
    page_icon="🎭",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Title and description
st.title("🎭 AttriDiffuser: Text-to-Face Generation")
st.markdown("""
This application implements a simplified version of **AttriDiffuser** from the research paper:
*"AttriDiffuser: Adversarially enhanced diffusion model for text-to-facial attribute image synthesis"*

Generate realistic face images from text descriptions with multiple attributes!
""")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'generated_image_paths' not in st.session_state:
    st.session_state.generated_image_paths = []
if 'clip_scores' not in st.session_state:
    st.session_state.clip_scores = []
if 'last_prompt' not in st.session_state:
    st.session_state.last_prompt = ""
if 'needs_clip' not in st.session_state:
    st.session_state.needs_clip = False

# Sidebar for settings
st.sidebar.header("⚙️ Settings")

# Performance tips
with st.sidebar.expander("⚡ Performance Tips"):
    st.markdown("""
    **Running on CPU - Tips for faster generation:**
    - Use 15-20 inference steps (default: 20)
    - Generate 1 image at a time
    - Lower guidance scale (7.0-7.5)
    - Close other applications
    
    **Estimated times (CPU):**
    - 15 steps: ~2-3 minutes
    - 20 steps: ~3-4 minutes
    - 30 steps: ~5-7 minutes
    
    💡 For faster generation, use a GPU-enabled system!
    """)

st.sidebar.markdown("---")

# Dataset path
dataset_path = st.sidebar.text_input(
    "Dataset ZIP Path",
    value=r"C:\Users\hp\Downloads\drive-download-20260312T133839Z-3-003.zip",
    help="Path to your local dataset ZIP file"
)

# Extract dataset button
if st.sidebar.button("📦 Extract Dataset"):
    with st.spinner("Extracting dataset..."):
        success = extract_dataset(dataset_path)
        if success:
            st.sidebar.success("✅ Dataset extracted successfully!")
        else:
            st.sidebar.error("❌ Failed to extract dataset")

# Model loading
st.sidebar.header("🤖 Model")
load_model = st.sidebar.button("Load Model")

if load_model:
    with st.spinner("Loading AttriDiffuser model... This may take a few minutes..."):
        try:
            st.session_state.model = SimplifiedAttriDiffuser()
            st.sidebar.success("✅ Model loaded!")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {e}")

# ── Run CLIP computation BEFORE columns (top-level, persists across reruns) ───
if st.session_state.needs_clip:
    # Load images from saved paths if PIL objects lost across rerun
    images_for_clip = st.session_state.generated_images
    if not images_for_clip and st.session_state.generated_image_paths:
        images_for_clip = [Image.open(p) for p in st.session_state.generated_image_paths if os.path.exists(p)]
    if images_for_clip and st.session_state.last_prompt:
        scores = []
        for img in images_for_clip:
            score = compute_clip_score(img, st.session_state.last_prompt)
            if score is not None:
                scores.append(score)
        st.session_state.clip_scores = scores
    st.session_state.needs_clip = False
# ─────────────────────────────────────────────────────────────────────────────

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Input")
    
    # Text input
    text_description = st.text_area(
        "Describe the face you want to generate:",
        placeholder="Example: A young female with a smiling expression, brown hair, and blue eyes",
        height=100,
        help="Describe facial attributes like age, gender, expression, hair color, etc."
    )
    
    # Attribute examples
    with st.expander("💡 Example Descriptions"):
        st.markdown("""
        - "A young male with a serious expression and black hair"
        - "An elderly female with a smiling face and gray hair"
        - "A middle-aged male with glasses and brown hair"
        - "A young female with long blonde hair and green eyes"
        - "An Asian male with short black hair and a neutral expression"
        """)
    
    # Generation parameters
    st.subheader("Generation Parameters")
    
    # Fast mode toggle
    fast_mode = st.checkbox(
        "⚡ Fast Mode (lower quality, faster generation)",
        value=False,
        help="Reduces image size to 256x256 and uses fewer steps for 2-3x faster generation"
    )
    
    if fast_mode:
        num_images = 1
        num_steps = 15
        st.info("⚡ Fast Mode: 1 image, 15 steps, 256x256 resolution (~1-2 min)")
    else:
        num_images = st.slider(
            "Number of images to generate",
            min_value=1,
            max_value=4,
            value=1,
            help="Generate multiple diverse faces from the same description"
        )
        
        num_steps = st.slider(
            "Quality (inference steps)",
            min_value=10,
            max_value=50,
            value=20,
            help="More steps = better quality but slower generation. 15-25 is recommended for CPU."
        )
    
    guidance_scale = st.slider(
        "Guidance scale",
        min_value=5.0,
        max_value=15.0,
        value=7.5,
        step=0.5,
        help="Higher values = closer to text description"
    )
    
    # Generate button
    generate_btn = st.button("🎨 Generate Face(s)", type="primary", use_container_width=True)

with col2:
    st.header("🖼️ Generated Results")
    
    if generate_btn:
        if st.session_state.model is None:
            st.warning("⚠️ Please load the model first using the sidebar!")
        elif not text_description.strip():
            st.warning("⚠️ Please enter a text description!")
        else:
            # Parse attributes
            attributes = parse_attributes(text_description)
            st.info(f"Detected attributes: {attributes}")
            st.session_state.last_prompt = text_description
            st.session_state.clip_scores = []
            st.session_state.needs_clip = False
            
            # Custom CSS for stylish progress bar
            st.markdown("""
                <style>
                .progress-container {
                    width: 100%;
                    background-color: #f0f2f6;
                    border-radius: 10px;
                    padding: 3px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 20px 0;
                }
                .progress-bar {
                    height: 30px;
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    border-radius: 8px;
                    transition: width 0.3s ease;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: bold;
                    font-size: 14px;
                }
                .progress-text {
                    text-align: center;
                    margin-top: 10px;
                    font-size: 16px;
                    color: #667eea;
                    font-weight: 600;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Generate images with progress tracking
            start_time = time.time()
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            timer_placeholder = st.empty()
            
            # Show initial progress
            progress_placeholder.markdown(f"""
                <div class="progress-container">
                    <div class="progress-bar" style="width: 0%;">
                        0%
                    </div>
                </div>
                <div class="progress-text">
                    Initializing generation...
                </div>
            """, unsafe_allow_html=True)
            
            if num_images == 1:
                # Single image generation with real-time progress
                status_placeholder.info("🎨 Generating your face image...")
                step_times = []
                
                def update_progress(step, total_steps):
                    current_time = time.time()
                    elapsed = current_time - start_time
                    
                    # Calculate average time per step
                    if step > 1:
                        step_times.append(elapsed / step)
                        avg_time_per_step = sum(step_times[-5:]) / len(step_times[-5:])  # Use last 5 steps
                        remaining_steps = total_steps - step
                        estimated_remaining = avg_time_per_step * remaining_steps
                    else:
                        estimated_remaining = 0
                    
                    progress = int((step / total_steps) * 100)
                    
                    # Format time
                    elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
                    remaining_str = time.strftime("%M:%S", time.gmtime(estimated_remaining)) if estimated_remaining > 0 else "calculating..."
                    
                    progress_placeholder.markdown(f"""
                        <div class="progress-container">
                            <div class="progress-bar" style="width: {progress}%;">
                                {progress}%
                            </div>
                        </div>
                        <div class="progress-text">
                            Processing step {step}/{total_steps}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    timer_placeholder.markdown(f"""
                        <div style="text-align: center; margin: 10px 0; font-size: 18px;">
                            <span style="color: #667eea; font-weight: bold;">⏱️ Elapsed: {elapsed_str}</span>
                            <span style="margin: 0 20px;">|</span>
                            <span style="color: #764ba2; font-weight: bold;">⏳ Remaining: ~{remaining_str}</span>
                        </div>
                    """, unsafe_allow_html=True)
                
                image = st.session_state.model.generate_face(
                    text_description,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    callback=update_progress,
                    fast_mode=fast_mode
                )
                
                # Complete progress
                final_time = time.time() - start_time
                final_time_str = time.strftime("%M:%S", time.gmtime(final_time))
                
                progress_placeholder.markdown(f"""
                    <div class="progress-container">
                        <div class="progress-bar" style="width: 100%;">
                            100%
                        </div>
                    </div>
                    <div class="progress-text">
                        Generation complete!
                    </div>
                """, unsafe_allow_html=True)
                
                timer_placeholder.markdown(f"""
                    <div style="text-align: center; margin: 10px 0; font-size: 18px;">
                        <span style="color: #28a745; font-weight: bold;">✅ Total Time: {final_time_str}</span>
                    </div>
                """, unsafe_allow_html=True)
                
                if image:
                    st.session_state.generated_images = [image]
            else:
                # Multiple images generation with real-time progress
                st.session_state.generated_images = []
                step_times = []
                
                for img_idx in range(num_images):
                    status_placeholder.info(f"🎨 Creating face variation {img_idx + 1}/{num_images}...")
                    
                    def update_progress(step, total_steps):
                        current_time = time.time()
                        elapsed = current_time - start_time
                        
                        # Calculate overall progress
                        total_steps_all = num_images * total_steps
                        completed_steps = (img_idx * total_steps) + step
                        
                        # Calculate average time per step
                        if completed_steps > 1:
                            step_times.append(elapsed / completed_steps)
                            avg_time_per_step = sum(step_times[-10:]) / len(step_times[-10:])
                            remaining_steps = total_steps_all - completed_steps
                            estimated_remaining = avg_time_per_step * remaining_steps
                        else:
                            estimated_remaining = 0
                        
                        # Calculate overall progress across all images
                        base_progress = (img_idx / num_images) * 100
                        current_image_progress = (step / total_steps) * (100 / num_images)
                        overall_progress = int(base_progress + current_image_progress)
                        
                        # Format time
                        elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
                        remaining_str = time.strftime("%M:%S", time.gmtime(estimated_remaining)) if estimated_remaining > 0 else "calculating..."
                        
                        progress_placeholder.markdown(f"""
                            <div class="progress-container">
                                <div class="progress-bar" style="width: {overall_progress}%;">
                                    {overall_progress}%
                                </div>
                            </div>
                            <div class="progress-text">
                                Face {img_idx + 1}/{num_images} - Step {step}/{total_steps}
                            </div>
                        """, unsafe_allow_html=True)
                        
                        timer_placeholder.markdown(f"""
                            <div style="text-align: center; margin: 10px 0; font-size: 18px;">
                                <span style="color: #667eea; font-weight: bold;">⏱️ Elapsed: {elapsed_str}</span>
                                <span style="margin: 0 20px;">|</span>
                                <span style="color: #764ba2; font-weight: bold;">⏳ Remaining: ~{remaining_str}</span>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    guidance = 7.5 + (img_idx * 0.5)
                    image = st.session_state.model.generate_face(
                        text_description,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance,
                        callback=update_progress,
                        fast_mode=fast_mode
                    )
                    if image:
                        st.session_state.generated_images.append(image)
                
                # Complete progress
                final_time = time.time() - start_time
                final_time_str = time.strftime("%M:%S", time.gmtime(final_time))
                
                progress_placeholder.markdown(f"""
                    <div class="progress-container">
                        <div class="progress-bar" style="width: 100%;">
                            100%
                        </div>
                    </div>
                    <div class="progress-text">
                        All faces generated!
                    </div>
                """, unsafe_allow_html=True)
                
                timer_placeholder.markdown(f"""
                    <div style="text-align: center; margin: 10px 0; font-size: 18px;">
                        <span style="color: #28a745; font-weight: bold;">✅ Total Time: {final_time_str}</span>
                    </div>
                """, unsafe_allow_html=True)
            
            elapsed_time = time.time() - start_time
            time.sleep(1.5)
            progress_placeholder.empty()
            status_placeholder.empty()
            timer_placeholder.empty()
            st.success(f"✅ Generated {len(st.session_state.generated_images)} face(s) in {elapsed_time:.2f} seconds!")
            
            # Save images to disk so they survive st.rerun()
            os.makedirs("generated_faces", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            paths = []
            for idx, img in enumerate(st.session_state.generated_images):
                path = f"generated_faces/face_{timestamp}_{idx+1}.png"
                img.save(path)
                paths.append(path)
            st.session_state.generated_image_paths = paths
            st.session_state.needs_clip = True
            st.rerun()
    
    # Display generated images
    display_images = st.session_state.generated_images
    if not display_images and st.session_state.generated_image_paths:
        display_images = [Image.open(p) for p in st.session_state.generated_image_paths if os.path.exists(p)]

    if display_images:
        if len(display_images) == 1:
            st.image(display_images[0], caption="Generated Face", width='stretch')
        else:
            cols = st.columns(2)
            for idx, img in enumerate(display_images):
                with cols[idx % 2]:
                    st.image(img, caption=f"Variation {idx+1}", width='stretch')

        # Download button
        if st.button("💾 Save Images"):
            os.makedirs("generated_faces", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_files = []
            for idx, img in enumerate(display_images):
                filename = f"generated_faces/face_{timestamp}_{idx+1}.png"
                img.save(filename)
                saved_files.append(filename)
            st.success(f"✅ Saved {len(display_images)} image(s)!")
            for filename in saved_files:
                st.text(f"📁 {filename}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CLIP Score Evaluation (always rendered)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("📊 CLIP Score Evaluation")
st.caption("Measures how well the generated image matches your text prompt. Computed automatically after each generation.")

if not st.session_state.clip_scores:
    st.info("Generate an image above to see the CLIP score evaluation here.")
else:
    for idx, score in enumerate(st.session_state.clip_scores):
        label = "Excellent 🟢" if score >= 0.28 else ("Good 🟡" if score >= 0.22 else "Low 🔴")
        c1, c2, c3 = st.columns([1, 5, 1])
        with c1:
            st.metric(f"Image {idx + 1}", f"{score:.3f}")
        with c2:
            st.progress(min(score / 0.35, 1.0))
        with c3:
            st.markdown(f"**{label}**")
    avg = round(sum(st.session_state.clip_scores) / len(st.session_state.clip_scores), 3)
    if avg >= 0.28:
        st.success(f"Average CLIP Score: **{avg}** — Great text-image alignment ✅")
    elif avg >= 0.22:
        st.warning(f"Average CLIP Score: **{avg}** — Moderate alignment. Try a more detailed prompt.")
    else:
        st.error(f"Average CLIP Score: **{avg}** — Low alignment. Be more specific in your description.")

    if st.session_state.last_prompt:
        st.caption(f"Evaluated against prompt: *\"{st.session_state.last_prompt}\"*")

# Footer
st.markdown("---")
st.markdown("""
### 📚 About AttriDiffuser
This project is based on the research paper published in Pattern Recognition (2025):
**"AttriDiffuser: Adversarially enhanced diffusion model for text-to-facial attribute image synthesis"**

Key Features:
- Multi-attribute facial control (age, gender, expression, etc.)
- High-fidelity face generation
- Diversity in generated faces
- Attribute-gating cross-attention mechanism

**Reference:** [ScienceDirect Article](https://www.sciencedirect.com/science/article/pii/S0031320325001074)
""")
