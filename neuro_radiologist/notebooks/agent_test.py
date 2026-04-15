import torch
from neuro_radiologist.app.llm_agent import MedicalReportGenerator

# Mocking the outputs we got from Phase 4
# Logits shape [1, 2], Heatmap shape [1, 4, 4, 4]
dummy_logits = torch.tensor([[0.02, -0.42]]) 
dummy_heatmap = torch.rand(1, 4, 4, 4)

# Initialize the generator
report_gen = MedicalReportGenerator()

# Generate and view the prompts
report_gen.mock_generate_reports(dummy_logits, dummy_heatmap)