# Build by Vaibhav Sonawane, contact: work.vaibhav1308@gmail.com

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# Load the processor and model from Hugging Face hub
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    torch_dtype=torch.float32, 

    # Set device_map="cuda" to enable GPU acceleration on systems with CUDA-compatible hardware for optimized performance.
    device_map="cpu" 
    
)

# Sample image URL
img_url = "https://cdn.pixabay.com/photo/2022/08/10/17/10/woman-7377662_960_720.jpg"

# Conversation-style input, as expected by multimodal models
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": img_url},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

# Tokenize and format the input conversation for the model
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# Generate output tokens using the model
output_ids = model.generate(**inputs, max_new_tokens=128)

# Decode the output tokens into human-readable text
generated_texts = processor.batch_decode(output_ids, skip_special_tokens=True)

# Print the generated description
print(generated_texts)
