import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import json
import os
from datetime import datetime
import io

# Import custom utilities
from utils.image_processor import ImageProcessor
from utils.ml_models import PlantHealthDetector, SoilHealthDetector
from utils.backup_system import BackupManager
from utils.recommendations import RecommendationEngine

# Initialize components
@st.cache_resource
def load_models():
    """Load ML models (cached for performance)"""
    plant_detector = PlantHealthDetector()
    soil_detector = SoilHealthDetector()
    return plant_detector, soil_detector

def main():
    st.set_page_config(
        page_title="AI Plant & Soil Health Detection",
        page_icon="🌱",
        layout="wide"
    )
    
    st.title("🌱 AI-Powered Plant & Soil Health Detection System")
    st.markdown("Upload images of plants or soil to get instant health analysis and recommendations")
    
    # Initialize session state
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Load models
    try:
        plant_detector, soil_detector = load_models()
        image_processor = ImageProcessor()
        backup_manager = BackupManager()
        recommendation_engine = RecommendationEngine()
    except Exception as e:
        st.error(f"Error loading system components: {str(e)}")
        st.stop()
    
    # Sidebar for settings and history
    with st.sidebar:
        st.header("Settings")
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Auto-detect", "Plant Health", "Soil Health"]
        )
        confidence_threshold = 0.5 
        # confidence_threshold = st.slider(
        #     "Confidence Threshold",
        #     min_value=0.0,
        #     max_value=1.0,
        #     value=0.5,
        #     step=0.1
        # )
        
        st.header("Analysis History")
        if st.button("Clear History"):
            st.session_state.analysis_history = []
            st.rerun()
        
        # Display recent analyses
        if st.session_state.analysis_history:
            for i, analysis in enumerate(st.session_state.analysis_history[-5:]):
                with st.expander(f"Analysis {len(st.session_state.analysis_history) - i}"):
                    st.write(f"**Type:** {analysis['type']}")
                    st.write(f"**Result:** {analysis['result']}")
                    st.write(f"**Confidence:** {analysis['confidence']:.2f}")
                    st.write(f"**Time:** {analysis['timestamp']}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Image Upload")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload clear images of plants or soil for best results"
        )
        
        if uploaded_file is not None:
            try:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Process image
                with st.spinner("Processing image..."):
                    processed_image = image_processor.preprocess_image(image)
                    
                    # Determine analysis type
                    if analysis_type == "Auto-detect":
                        detected_type = image_processor.detect_image_type(processed_image)
                        st.info(f"Auto-detected: {detected_type}")
                    else:
                        detected_type = analysis_type.replace(" Health", "").lower()
                    
                    # Perform analysis
                    if detected_type == "plant":
                        result = plant_detector.analyze(processed_image)
                        analysis_result = {
                            'type': 'Plant Health',
                            'result': result['condition'],
                            'confidence': result['confidence'],
                            'details': result['details']
                        }
                    elif detected_type == "soil":
                        result = soil_detector.analyze(processed_image)
                        analysis_result = {
                            'type': 'Soil Health',
                            'result': result['condition'],
                            'confidence': result['confidence'],
                            'details': result['details']
                        }
                    else:
                        st.error("Could not determine image type. Please select manually.")
                        st.stop()
                    
                    # Add timestamp
                    analysis_result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    analysis_result['image_name'] = uploaded_file.name
                    
                    # Store in history
                    st.session_state.analysis_history.append(analysis_result)
                    
                    # Backup analysis
                    backup_manager.backup_analysis(analysis_result, uploaded_file)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.stop()
    
    with col2:
        st.header("📊 Analysis Results")
        
        if uploaded_file is not None:
            try:
                if 'analysis_result' in locals() and analysis_result is not None:
                    # Display results
                    if analysis_result['confidence'] >= confidence_threshold:
                        if analysis_result['result'] in ['Healthy', 'Good']:
                            st.success(f"✅ **{analysis_result['type']}**: {analysis_result['result']}")
                        elif analysis_result['result'] in ['Moderate Issues', 'Fair']:
                            st.warning(f"⚠️ **{analysis_result['type']}**: {analysis_result['result']}")
                        else:
                            st.error(f"❌ **{analysis_result['type']}**: {analysis_result['result']}")
                        
                        # Confidence score
                        st.metric("Confidence Score", f"{analysis_result['confidence']:.2%}")
                        
                        # Detailed analysis
                        st.subheader("Detailed Analysis")
                        for key, value in analysis_result['details'].items():
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                        
                        # Recommendations
                        st.subheader("🎯 Recommendations")
                        recommendations = recommendation_engine.get_recommendations(analysis_result)
                        
                        for i, rec in enumerate(recommendations, 1):
                            st.write(f"{i}. {rec}")
                        
                    else:
                        st.warning(f"Low confidence result ({analysis_result['confidence']:.2%}). Consider uploading a clearer image.")
                else:
                    st.info("Analysis in progress...")
            except NameError:
                st.info("Analysis not yet available...")
        else:
            st.info("Upload an image to see analysis results here.")
    
    # Statistics and backup status
    st.header("📈 System Statistics")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Total Analyses", len(st.session_state.analysis_history))
    
    with col4:
        plant_analyses = len([a for a in st.session_state.analysis_history if a['type'] == 'Plant Health'])
        st.metric("Plant Analyses", plant_analyses)
    
    with col5:
        soil_analyses = len([a for a in st.session_state.analysis_history if a['type'] == 'Soil Health'])
        st.metric("Soil Analyses", soil_analyses)
    
    # Backup status
    backup_status = backup_manager.get_backup_status()
    st.info(f"🔄 Backup Status: {backup_status['status']} | Last Backup: {backup_status['last_backup']}")
    
    # Export functionality
    if st.session_state.analysis_history:
        if st.button("📥 Export Analysis History"):
            df = pd.DataFrame(st.session_state.analysis_history)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"plant_soil_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
