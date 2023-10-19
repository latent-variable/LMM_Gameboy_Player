from transformers import FuyuForCausalLM, AutoTokenizer, FuyuProcessor, FuyuImageProcessor
from PIL import Image
import torch
print('hi')
# load model, tokenizer, and processor
pretrained_path = "./fuyu-8b"
tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

image_processor = FuyuImageProcessor()
processor = FuyuProcessor(image_processor=image_processor, tokenizer=tokenizer)

print('Loading model')
model = FuyuForCausalLM.from_pretrained(pretrained_path, device_map="cuda:0", torch_dtype=torch.float16)

# test inference
text_prompt = "Generate a coco-style caption.\n"
image_path = "./fuyu-8b/bus.png"  # https://huggingface.co/adept-hf-collab/fuyu-8b/blob/main/bus.png
image_pil = Image.open(image_path)

model_inputs = processor(text=text_prompt, images=[image_pil], device="cuda:0")
for k, v in model_inputs.items():
    model_inputs[k] = v.to(device="cuda:0")

generation_output = model.generate(**model_inputs, max_new_tokens=7)
generation_text = processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)
print(generation_text)
assert generation_text == ['A bus parked on the side of a road.']