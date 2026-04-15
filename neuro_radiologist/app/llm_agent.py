import torch
import torch.nn.functional as F

class MedicalReportGenerator:
    """
    Translates 3D ViT outputs into dual-tier medical reports using an LLM.
    """
    def __init__(self, api_key=None, model_name="gpt-4"): # Placeholder for actual API setup
        self.api_key = api_key
        self.model_name = model_name
        # In a real scenario, you would initialize your OpenAI/Gemini client here

    def _process_model_outputs(self, logits, heatmap):
        """Converts raw tensors into readable metrics for the LLM."""
        # 1. Calculate Confidence via Softmax
        probabilities = F.softmax(logits, dim=-1)
        confidence = torch.max(probabilities).item() * 100
        predicted_class = torch.argmax(probabilities).item()
        
        diagnosis = "Anomaly Detected (Potential Lesion/Tumor)" if predicted_class == 1 else "No Anomalies Detected (Healthy)"
        
        # 2. Extract highest attention region (simplistic approach for the prompt)
        # We find the flattened index of the maximum attention weight
        flat_index = torch.argmax(heatmap.view(-1)).item()
        
        return diagnosis, confidence, flat_index

    def generate_prompts(self, logits, heatmap):
        diagnosis, confidence, focus_idx = self._process_model_outputs(logits, heatmap)
        
        # --- Prompt 1: The Clinician Report ---
        clinician_prompt = f"""
        Act as an expert Neuro-Radiologist. You are reviewing the outputs of an AI 3D Vision Transformer diagnostic system.
        
        System Outputs:
        - Primary Finding: {diagnosis}
        - AI Confidence Score: {confidence:.1f}%
        - Primary Attention Focus Index (Patch/Region): {focus_idx} (out of 64 total volumetric regions)
        
        Draft a brief, highly technical radiology report containing exactly two sections:
        1. FINDINGS: (Describe the AI's technical output)
        2. IMPRESSION: (Provide a clear, clinical summary)
        """
        
        # --- Prompt 2: The Patient Report ---
        patient_prompt = f"""
        Act as an empathetic, clear-spoken doctor explaining an AI scan result to a patient. 
        
        System Outputs:
        - Primary Finding: {diagnosis}
        - AI Confidence Score: {confidence:.1f}%
        
        Draft a short, reassuring explanation of these results. Do NOT use dense medical jargon. 
        Focus on clarity, compassion, and explaining what a {confidence:.1f}% AI confidence means in simple terms.
        Always advise them that an AI is a tool, and their human doctor will make the final decision.
        """
        
        return clinician_prompt, patient_prompt

    def mock_generate_reports(self, logits, heatmap):
        """
        A mock function to test the pipeline before spending money on API calls.
        """
        clinician_prompt, patient_prompt = self.generate_prompts(logits, heatmap)
        
        # In production, you pass these prompts to your LLM API.
        # Here, we just print the prompts to ensure the architecture is sound.
        print("\n" + "="*50)
        print("CLINICIAN API PROMPT PAYLOAD:")
        print("="*50)
        print(clinician_prompt)
        
        print("\n" + "="*50)
        print("PATIENT API PROMPT PAYLOAD:")
        print("="*50)
        print(patient_prompt)