import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.segmentation import mark_boundaries
import io
import base64

# Configure page
st.set_page_config(
    page_title="🔬 Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .info-box {
        background: #f8f9fa;
        border-left: 5px solid #2E86AB;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .stProgress .st-bp {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model('best_dr_model.h5')
        return model
    except:
        st.error("❌ Model not found! Please train the model first using the training script.")
        return None

class DRPredictor:
    def __init__(self):
        self.model = load_model()
        self.image_size = (384, 384)
        self.class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
        self.class_descriptions = {
            'No DR': 'No signs of diabetic retinopathy detected.',
            'Mild DR': 'Mild non-proliferative diabetic retinopathy. Small areas of balloon-like swelling in retinal blood vessels.',
            'Moderate DR': 'Moderate non-proliferative diabetic retinopathy. Blood vessels are blocked, depriving portions of the retina.',
            'Severe DR': 'Severe non-proliferative diabetic retinopathy. Many blood vessels are blocked, areas of retina are deprived of blood supply.',
            'Proliferative DR': 'Proliferative diabetic retinopathy. Most advanced stage with new blood vessel growth and potential vision loss.'
        }
        self.class_colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#6f42c1']
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Resize
        img_resized = cv2.resize(img_array, self.image_size)
        
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch, img_resized
    
    def predict(self, image):
        """Make prediction on image"""
        if self.model is None:
            return None
        
        img_batch, img_processed = self.preprocess_image(image)
        
        # Prediction
        prediction = self.model.predict(img_batch, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        return {
            'predicted_class': predicted_class,
            'class_name': self.class_names[predicted_class],
            'confidence': confidence,
            'all_probabilities': prediction[0],
            'processed_image': img_processed
        }
    
    def explain_prediction(self, image, prediction_result):
        """Generate explanation using LIME"""
        if self.model is None:
            return None
        
        try:
            img_array = np.array(image.resize(self.image_size))
            
            def predict_fn(images):
                processed = images.astype(np.float32) / 255.0
                return self.model.predict(processed, verbose=0)
            
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                img_array,
                predict_fn,
                top_labels=2,
                hide_color=0,
                num_samples=1000
            )
            
            temp, mask = explanation.get_image_and_mask(
                prediction_result['predicted_class'],
                positive_only=True,
                num_features=5,
                hide_rest=False
            )
            
            # Create explanation visualization
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(mark_boundaries(temp / 255.0, mask))
            ax.set_title(f'Important Regions for {prediction_result["class_name"]} Prediction')
            ax.axis('off')
            
            # Convert to base64 for display
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode()
            plt.close()
            
            return img_b64
        except Exception as e:
            st.warning(f"Could not generate explanation: {str(e)}")
            return None

def create_probability_chart(probabilities, class_names, class_colors):
    """Create interactive probability chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=probabilities,
            marker_color=class_colors,
            text=[f'{p:.3f}' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Diabetic Retinopathy Classes",
        yaxis_title="Probability",
        height=400,
        template="plotly_white"
    )
    
    return fig

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Diabetic Retinopathy Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced AI-powered retinal image analysis for diabetic retinopathy screening</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
    # Model info
    predictor = DRPredictor()
    
    if predictor.model is not None:
        st.sidebar.success("✅ Model loaded successfully!")
        st.sidebar.info(f"📊 Model Parameters: {predictor.model.count_params():,}")
    else:
        st.sidebar.error("❌ Model not loaded")
        st.stop()
    
    # File upload
    st.sidebar.header("📤 Upload Retinal Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a retinal fundus image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear retinal fundus photograph for analysis"
    )
    
    # Settings
    st.sidebar.header("⚙️ Settings")
    show_explanation = st.sidebar.checkbox("Show AI Explanation (LIME)", value=True)
    show_probabilities = st.sidebar.checkbox("Show All Class Probabilities", value=True)
    
    # Main content
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(image, caption="Uploaded Retinal Image", use_column_width=True)
            
            # Image info
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.write(f"**Image Dimensions:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**File Size:** {len(uploaded_file.getvalue()) / 1024:.1f} KB")
            st.write(f"**Format:** {image.format}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("🤖 AI Analysis")
            
            # Progress bar for prediction
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Processing image...")
            progress_bar.progress(25)
            
            # Make prediction
            result = predictor.predict(image)
            progress_bar.progress(75)
            
            if result:
                status_text.text("Analysis complete!")
                progress_bar.progress(100)
                
                # Display prediction
                confidence_color = predictor.class_colors[result['predicted_class']]
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>🎯 Diagnosis: {result['class_name']}</h2>
                    <h3>📊 Confidence: {result['confidence']:.1%}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Description
                description = predictor.class_descriptions[result['class_name']]
                st.markdown(f"""
                <div class="info-box">
                    <strong>Description:</strong> {description}
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence metrics
                col_metrics = st.columns(3)
                with col_metrics[0]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{result['confidence']:.1%}</h3>
                        <p>Confidence</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_metrics[1]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{result['predicted_class']}</h3>
                        <p>Severity Level</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_metrics[2]:
                    risk_level = "Low" if result['predicted_class'] <= 1 else "High" if result['predicted_class'] >= 3 else "Medium"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{risk_level}</h3>
                        <p>Risk Level</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Additional analysis sections
        if show_probabilities and result:
            st.subheader("📊 Detailed Probability Analysis")
            
            # Probability chart
            fig = create_probability_chart(
                result['all_probabilities'], 
                predictor.class_names, 
                predictor.class_colors
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Probability table
            prob_df = {
                'Class': predictor.class_names,
                'Probability': [f"{p:.4f}" for p in result['all_probabilities']],
                'Percentage': [f"{p:.2%}" for p in result['all_probabilities']]
            }
            st.table(prob_df)
        
        if show_explanation and result:
            st.subheader("🧠 AI Explanation (LIME)")
            
            with st.spinner("Generating explanation... This may take a moment."):
                explanation_img = predictor.explain_prediction(image, result)
                
                if explanation_img:
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <img src="data:image/png;base64,{explanation_img}" style="max-width: 100%; height: auto;"/>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info("🔍 The highlighted regions show areas the AI model focused on when making its prediction. Warmer colors indicate higher importance.")
        
        # Clinical recommendations
        st.subheader("⚕️ Clinical Recommendations")
        
        if result['predicted_class'] == 0:
            st.success("✅ **No immediate action required.** Continue regular diabetic care and annual eye exams.")
        elif result['predicted_class'] == 1:
            st.warning("⚠️ **Mild DR detected.** Recommend follow-up with ophthalmologist within 6-12 months.")
        elif result['predicted_class'] == 2:
            st.warning("⚠️ **Moderate DR detected.** Ophthalmologist consultation recommended within 3-6 months.")
        elif result['predicted_class'] == 3:
            st.error("🚨 **Severe DR detected.** Urgent ophthalmologist referral recommended within 1-2 months.")
        else:
            st.error("🚨 **Proliferative DR detected.** Immediate ophthalmologist consultation required!")
        
        st.info("ℹ️ **Disclaimer:** This AI system is for screening purposes only and should not replace professional medical diagnosis. Please consult with a qualified ophthalmologist for definitive diagnosis and treatment.")
    
    else:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to DR Detection System
        
        This advanced AI system uses deep learning to analyze retinal fundus photographs and detect diabetic retinopathy at various stages.
        
        ### 🎯 Features:
        - **High Accuracy:** State-of-the-art EfficientNetV2L architecture
        - **5-Class Classification:** No DR, Mild, Moderate, Severe, Proliferative
        - **AI Explainability:** LIME-based visual explanations
        - **Clinical Recommendations:** Evidence-based follow-up suggestions
        
        ### 📤 How to Use:
        1. Upload a high-quality retinal fundus image using the sidebar
        2. Wait for the AI analysis to complete
        3. Review the diagnosis, confidence score, and recommendations
        4. Optionally view detailed explanations and probability distributions
        
        ### ⚠️ Important Notes:
        - This system is for screening purposes only
        - Always consult with a qualified ophthalmologist
        - Ensure images are clear and well-focused
        - Best results with standard fundus photography
        """)
        
        # Sample images section
        st.subheader("🖼️ Sample Images")
        st.info("💡 **Tip:** For best results, upload clear retinal fundus photographs with good contrast and minimal artifacts.")

if __name__ == "__main__":
    main()