
# VisionCore <h2>🧠SmolVLM2 Image Description Script</h2>

**Author:** [Vaibhav Sonawane](mailto:work.vaibhav1308@gmail.com)  
**Model Used:** `SmolVLM2-500M-Video-Instruct` by HuggingFaceTB  
**Frameworks:** PyTorch, Transformers (Hugging Face) <br>
**Reference**: Xuan-Son Nguyen

---

## 📌 Overview

This script uses the **SmolVLM2-256M-Video-Instruct** model from Hugging Face to **generate natural language descriptions of images**. It loads the model and processor, fetches an image from a URL, and processes it using the `image-to-text` capability of the model.

---

## 🚀 Features

- Use **SmolVLM2**, a compact vision-language model.
- Accepts **image + prompt** and generates text description.
- Designed to run on **CPU or GPU (CUDA supported)**.
- Simple, minimal, and ready-to-integrate into larger applications.

---

## 🔧 Requirements

Install the following dependencies:

```bash
pip install torch transformers
```

---

## ▶️ How to Run

1. Make sure Python 3.8+ and pip are installed.
2. Save the script below as `image_processing_model.py`
3. Run the script:

```bash
python image_processing_model.py
```

You will see the image description printed in your terminal.

---

## 📄 Code Explanation (With Inline Documentation)

```python
# Build by Vaibhav Sonawane, contact: work.vaibhav1308@gmail.com

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# Load the processor and model from Hugging Face hub
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    torch_dtype=torch.float32, 
    device_map="cpu"  # Change to "cuda" if running on GPU
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
```

---

## 📷 Example Output

```bash
['The image shows a woman wearing a hat and smiling outdoors, possibly during sunset or golden hour.']
```

---

## ⚙️ Optional: Enable GPU

To enable CUDA acceleration, replace:

```python
device_map="cpu"
```

with:

```python
device_map="cuda"
```

Make sure you have a CUDA-compatible GPU and PyTorch installed with CUDA support.

---

## 📌 Credits

- **Model:** [SmolVLM2-256M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct)
- **Framework:** [Hugging Face Transformers](https://github.com/huggingface/transformers)
- **Image Source:** [Pixabay](https://pixabay.com/)
