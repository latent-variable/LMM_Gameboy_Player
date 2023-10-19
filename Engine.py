
import torch
import sys
import os
import re
import random
import time 
import numpy as np

# Allow LLaVA to locate it's dependencies 
LLaVA_Path = os.path.abspath('./LLaVA/')
sys.path.append(LLaVA_Path)
print(LLaVA_Path)

from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


from transformers import FuyuForCausalLM, AutoTokenizer, FuyuProcessor, FuyuImageProcessor


from prompt import get_gameplay_prompt

# from PIL import Image
from PIL import Image
import requests
from io import BytesIO

from transformers import TextStreamer

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image



    
class DummyBrain:
    def __init__(self):
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'SELECT', 'START']

    def get_action(self, image):
        # Pick a random action from the list
        # time.sleep(5)
        return random.choice(self.actions)




class DummyConfig:
    def __init__(self, image_aspect_ratio='pad'):
        self.image_aspect_ratio = image_aspect_ratio

class LLaVABrain:
    def __init__(self, model_path, device='cuda', temperature=0.2, max_new_tokens=128, load_8bit=True, load_4bit=False):
        disable_torch_init()
        # Initialization code from main() here
        print(f'Loading {model_path}')
        model_name = get_model_name_from_path(model_path)
        print(f'model_name: {model_name}')
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.conv_mode = self.get_conv_mode(model_name)  
        self.conv = conv_templates[self.conv_mode].copy()
        self.roles = ('user', 'assistant')  # Or whatever roles are correct for your template
        

    def get_conv_mode(self, model_name):
        conv_mode = None 
        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
            
        print(f'conv_mode: {conv_mode}')
        return conv_mode

    def get_action(self, image_file):
        self.conv = conv_templates[self.conv_mode].copy()
   

        image = load_image(image_file)
        model_cfg = DummyConfig(image_aspect_ratio='pad')  # or 'crop', based on your needs
        image_tensor = process_images([image], self.image_processor, model_cfg)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        text_input = get_gameplay_prompt()
        default_inp = f"{self.roles[0]}: {text_input}"
        inp = DEFAULT_IMAGE_TOKEN + '\n' + default_inp
        self.conv.append_message(self.conv.roles[0], inp) # append user
        self.conv.append_message(self.conv.roles[1], None) # append assistant

       
        prompt = self.conv.get_prompt()
        # print('\n\n**prompt', prompt)

    
        # Process image 
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        # streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Get LLaVA response 
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                # streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        self.remove_user_tokens_from_history()

        self.conv.messages[-1][-1] = outputs

       
        print(f'\n\n outputs: {outputs} \n\n')

        # Here, parse 'outputs' to extract the game action.
        #action = self.grab_action_from_response(outputs)

        return None
    
    def remove_user_tokens_from_history(self ):
        clear_user_inp = f"{self.roles[0]}: "
        self.conv.messages[-2][-1] = clear_user_inp

    def grab_action_from_response(self, text_data):
        valid_options = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'SELECT', 'START']
        # Using string methods
        for option in valid_options:
            button_option = option +' button'
            if button_option in text_data:
                print(f"Found option using string method: {option}")
                return option
                

        # Using regular expression
        pattern = r'\b(?:' + '|'.join(valid_options) + r')\b'
        match = re.search(pattern, text_data)
        if match:
            print(f"Found option using regex: {match.group()}")
            return match.group()
        
        print('No matching valid input found!')
        return None 




class FuyuBrain:
    def __init__(self, model_path,  temperature=0.3, max_new_tokens=128):
        # load model, tokenizer, and processor
        pretrained_path = model_path 
        tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        image_processor = FuyuImageProcessor()
        self.processor = FuyuProcessor(image_processor=image_processor, tokenizer=tokenizer)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print('Loading Fuyu model')
        self.model = FuyuForCausalLM.from_pretrained(pretrained_path, device_map="cuda:0", torch_dtype=torch.float16)
        


    def get_action(self, image_file):

        image_pil = Image.open(image_file)
        # test inference
        text_prompt = get_gameplay_prompt()
        # text_prompt = "Generate a coco-style caption.\n"
        
    
        model_inputs = self.processor(text=text_prompt, images=[image_pil], device="cuda:0")
        for k, v in model_inputs.items():
            model_inputs[k] = v.to(device="cuda:0")

        generation_output = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens, temperature=self.temperature)
        generation_text = self.processor.batch_decode(generation_output, skip_special_tokens=True)[0]
   
        img_tokens, text_output = generation_text.split('<s>')
        text_output = text_output.replace(text_prompt, "")
        print(f'\n\n outputs: {text_output} \n\n')

        # Here, parse 'outputs' to extract the game action.
        action = self.grab_action_from_response(text_output)

        return action
    
    def remove_user_tokens_from_history(self ):
        clear_user_inp = f"{self.roles[0]}: "
        self.conv.messages[-2][-1] = clear_user_inp

    def grab_action_from_response(self, text_data):
        valid_options = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'SELECT', 'START']
        # Using string methods
        for option in valid_options:
            button_option = option +' button'
            if button_option in text_data:
                print(f"Found option using string method: {option}")
                return option
                
        # Using regular expression
        pattern = r'\b(?:' + '|'.join(valid_options) + r')\b'
        match = re.search(pattern, text_data)
        if match:
            print(f"Found option using regex: {match.group()}")
            return match.group()
        
        print('No matching valid input found!')
        return None 