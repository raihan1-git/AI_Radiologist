import os
import torch
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    print("WARNING: GEMINI_API_KEY not found in .env file.")

print(api_key)  # Debugging line to confirm the API key is loaded

class MedicalReportGenerator:
    def __init__(self):
        # We use gemini-1.5-flash for rapid, highly capable text generation
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def generate_reports(self, logits, heatmap_3d):
        """
        Takes the PyTorch outputs, formats the context, and queries the LLM
        for both the Clinician and Patient reports.
        """
        # 1. Extract Clinical Metrics from Tensors
        probabilities = torch.softmax(logits, dim=-1)[0]
        confidence_score = probabilities.max().item() * 100
        predicted_class = torch.argmax(probabilities).item()
        diagnosis = "Anomaly Detected" if predicted_class == 1 else "No Anomaly Detected"
        
        # 2. Extract Spatial Coordinates from Heatmap
        # Find the highest attention concentration in the 4x4x4 grid
        max_idx = np.unravel_index(np.argmax(heatmap_3d, axis=None), heatmap_3d.shape)
        
        # 3. Draft the Base Payload
        clinical_context = f"""
        System Outputs:
        - Primary Finding: {diagnosis}
        - AI Confidence Score: {confidence_score:.2f}%
        - Primary Attention Focus (Z, Y, X coordinate block): {max_idx}
        """

        # 4. Generate Clinician Report
        clinician_prompt = f"""
        Act as an expert Neuro-Radiologist. Review the following AI system outputs for an MRI scan:
        {clinical_context}
        
        Draft a brief, highly technical diagnostic report (max 3 sentences) structured as:
        1. FINDINGS: (Describe what the AI's coordinates and confidence suggest).
        2. IMPRESSION: (Provide the clinical conclusion).
        """
        
        # 5. Generate Patient Report
        patient_prompt = f"""
        Act as an empathetic, clear-communicating doctor. Review the following AI system outputs:
        {clinical_context}
        
        Draft a short, reassuring explanation (max 3 sentences) for the patient. 
        Avoid dense medical jargon. Explain what the AI looked at and what the general outcome means. 
        Remind them this is an AI tool and a human doctor will make the final call.
        """

        # 6. Execute the API Calls
        try:
            clinician_response = self.model.generate_content(clinician_prompt)
            patient_response = self.model.generate_content(patient_prompt)
            
            return clinician_response.text, patient_response.text
        except Exception as e:
            return f"API Error: {str(e)}", f"API Error: {str(e)}"