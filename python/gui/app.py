import streamlit as st
import sys
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.rpca.mog_rpca import run_alg
from utils.video_utils import process_video_with_algorithm


st.set_page_config(
    page_title="MoG-RPCA Video Processing",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 MoG-RPCA: Mixture of Gaussians Robust PCA")
st.markdown("*Robust Principal Component Analysis with Complex Noise*")

st.sidebar.header("Algorithm Information")
st.sidebar.info("""
**Category:** RPCA

**Algorithm:** MoG-RPCA

**Speed Class:** 2 (2-5 seconds)

**Description:** Uses a Mixture of Gaussians model with Bayesian inference for robust PCA. 
Decomposes video into low-rank background (L) and sparse foreground (S) components.
""")

st.sidebar.header("Algorithm Parameters")

with st.sidebar.expander("MoG Parameters", expanded=True):
    mog_k = st.slider("Number of Gaussians (mog_k)", 1, 10, 3, 
                      help="Number of Gaussian components in the mixture model")

with st.sidebar.expander("Low-Rank Parameters", expanded=True):
    r = st.slider("Base rank (r)", 1, 10, 1,
                  help="Base rank for initial_rank = 2*r")
    lr_init = st.selectbox("Initialization Method", ['SVD', 'rand'],
                          help="Method for initializing low-rank component")

with st.sidebar.expander("Optimization Parameters", expanded=False):
    maxiter = st.number_input("Max Iterations", 1, 500, 100,
                              help="Maximum number of iterations")
    tol = st.number_input("Tolerance", 1e-6, 1e-1, 1e-3, format="%.1e",
                         help="Convergence tolerance")

with st.sidebar.expander("Prior Hyperparameters", expanded=False):
    st.markdown("**Low-Rank Priors:**")
    lr_a0 = st.number_input("a0", 1e-10, 1e-1, 1e-6, format="%.1e")
    lr_b0 = st.number_input("b0", 1e-10, 1e-1, 1e-6, format="%.1e")
    
    st.markdown("**MoG Priors:**")
    mog_mu0 = st.number_input("mu0", -1.0, 1.0, 0.0)
    mog_c0 = st.number_input("c0", 1e-10, 1e-1, 1e-3, format="%.1e")
    mog_d0 = st.number_input("d0", 1e-10, 1e-1, 1e-3, format="%.1e")
    mog_alpha0 = st.number_input("alpha0", 1e-10, 1e-1, 1e-3, format="%.1e")
    mog_beta0 = st.number_input("beta0", 1e-10, 1e-1, 1e-3, format="%.1e")

st.header("Video Upload")
uploaded_file = st.file_uploader(
    "Upload a video file (AVI or MP4)",
    type=['avi', 'mp4'],
    help="Upload a video to decompose into background and foreground components"
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_video_path = tmp_file.name
    
    st.success(f"Uploaded: {uploaded_file.name}")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.video(tmp_video_path)
        st.caption("Original Video")
    
    if st.button("🚀 Process Video", type="primary"):
        params = {
            'mog_k': mog_k,
            'r': r,
            'lr_init': lr_init,
            'maxiter': maxiter,
            'tol': tol,
            'lr_a0': lr_a0,
            'lr_b0': lr_b0,
            'mog_mu0': mog_mu0,
            'mog_c0': mog_c0,
            'mog_d0': mog_d0,
            'mog_alpha0': mog_alpha0,
            'mog_beta0': mog_beta0
        }
        
        with st.spinner('Processing video with MoG-RPCA algorithm...'):
            try:
                output_dir = Path(tempfile.mkdtemp())
                
                video_results = process_video_with_algorithm(
                    tmp_video_path,
                    run_alg,
                    params,
                    output_dir
                )
                
                st.success("✅ Processing complete!")
                
                estimated_rank = video_results['results']['O']['estimated_rank']
                st.info(f"**Estimated Rank:** {estimated_rank}")
                
                st.header("Results")
                st.markdown("*View the decomposed components side-by-side*")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.subheader("Input (I)")
                    st.video(video_results['input'])
                    st.caption("Original video")
                
                with col2:
                    st.subheader("Output (O)")
                    st.video(video_results['output'])
                    st.caption("Reconstructed background")
                
                with col3:
                    st.subheader("Low-rank (L)")
                    st.video(video_results['L'])
                    st.caption("Background component")
                
                with col4:
                    st.subheader("Sparse (S)")
                    st.video(video_results['S'])
                    st.caption("Foreground component")
                
                with st.expander("Download Results"):
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        with open(video_results['L'], 'rb') as f:
                            st.download_button(
                                "📥 Download Low-rank (L)",
                                f,
                                file_name=f"{Path(uploaded_file.name).stem}_low_rank.mp4",
                                mime="video/mp4"
                            )
                    
                    with col_dl2:
                        with open(video_results['S'], 'rb') as f:
                            st.download_button(
                                "📥 Download Sparse (S)",
                                f,
                                file_name=f"{Path(uploaded_file.name).stem}_sparse.mp4",
                                mime="video/mp4"
                            )
                
                shutil.rmtree(output_dir, ignore_errors=True)
                
            except Exception as e:
                st.error(f"❌ Error processing video: {str(e)}")
                st.exception(e)
    
    Path(tmp_video_path).unlink(missing_ok=True)

else:
    st.info("👆 Please upload a video file to get started")

st.markdown("---")
st.markdown("""

**Reference:** Qian Zhao, Deyu Meng, Zongben Xu, Wangmeng Zuo, Lei Zhang. 
"Robust Principal Component Analysis with Complex Noise." ICML, 2014.

**Key Features:**
- Bayesian inference with Mixture of Gaussians model
- Automatic rank determination
- Handles complex noise patterns
- SVD-based initialization for fast convergence
""")
