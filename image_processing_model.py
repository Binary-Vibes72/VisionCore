# Build by Vaibhav Sonawane, contact: @work.vaibhav1308@gmail.com
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    torch_dtype=torch.float32, 
    device_map="cpu" 
    # Set device_map="cuda" to enable GPU acceleration on systems with CUDA-compatible hardware for optimized performance.
)

img_url = "https://cdn.pixabay.com/photo/2022/08/10/17/10/woman-7377662_960_720.jpg"

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": img_url},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=128)

generated_texts = processor.batch_decode(output_ids, skip_special_tokens=True)
print(generated_texts)
