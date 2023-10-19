import os
import time 

import numpy as np
from PIL import Image
from pyboy import PyBoy, WindowEvent

import Engine 
from threading import Lock, Thread


class Env:
    def __init__(self, rom_path):
       
        self.frame_count = 0
        self.action_space = [WindowEvent.PRESS_ARROW_UP, 
                             WindowEvent.PRESS_ARROW_DOWN,
                             WindowEvent.PRESS_ARROW_LEFT,
                             WindowEvent.PRESS_ARROW_RIGHT,
                             WindowEvent.PRESS_BUTTON_A,
                             WindowEvent.PRESS_BUTTON_B,
                             WindowEvent.PRESS_BUTTON_SELECT,
                             WindowEvent.PRESS_BUTTON_START]
        

        # self.brain = Engine.DummyBrain()
        # self.brain = Engine.LLaVABrain(model_path='./LLaVA/liuhaotian/llava-v1.5-13b')
        self.brain = Engine.FuyuBrain(model_path="./fuyu-8b")

        # Laynch game
        self.pyboy = PyBoy(rom_path)

        self.state_path = rom_path +'.state'

        # pick up where you left off 
        if os.path.exists(self.state_path):
           # Load file
           file_like_object = open(self.state_path, "rb")
           self.pyboy.load_state(file_like_object)

        # action-lock 
        self.action_lock = Lock()

    def reset(self):
        self.pyboy.stop(save=False)
        self.pyboy = PyBoy(self.pyboy.cartridge)
        self.frame_count = 0

    def save_image(self, image, file_path, upscale_factor=2):
        """Save a PIL Image object to the specified file path."""
        width, height = image.size
        new_size = (width * upscale_factor, height * upscale_factor)
        image_resized = image.resize(new_size, 1)
        image_resized.save(file_path)

    def map_action_to_button(self, action):
        action_map = {
                'UP': WindowEvent.PRESS_ARROW_UP,
                'DOWN': WindowEvent.PRESS_ARROW_DOWN,
                'LEFT': WindowEvent.PRESS_ARROW_LEFT,
                'RIGHT': WindowEvent.PRESS_ARROW_RIGHT,
                'A': WindowEvent.PRESS_BUTTON_A,
                'B': WindowEvent.PRESS_BUTTON_B,
                'SELECT': WindowEvent.PRESS_BUTTON_SELECT,
                'START': WindowEvent.PRESS_BUTTON_START,
        }
        return action_map.get(action, None)
    
    def step(self):
     
        self.frame_count += 1
        window_closed = self.pyboy.tick()

        if window_closed:
            with open(self.state_path, "wb") as file_like_object:
                self.pyboy.save_state(file_like_object)
            return True
        
        # self.step_function()
        # Check if the lock is acquired. If not, acquire it and proceed.
        if self.action_lock.acquire(False):  # False ensures it won't block if the lock is already acquired.
            step_thread = Thread(target=self.step_function)
            step_thread.start()
            # step_thread.join(timeout=5)  # Wait for 1 second
            # if step_thread.is_alive():
            #     print("Warning: step_function thread is hanging.")
       
    def step_function(self):
        try:
            screen = self.pyboy.screen_image()
            image = Image.fromarray(np.array(screen))
            self.save_image(image, './frame.png')
            time.sleep(.5)
            action = self.brain.get_action('./frame.png')
            mapped_action = self.map_action_to_button(action)
            if mapped_action:
                print(f'Action: {action}, mapped_action: {mapped_action}')
                self.pyboy.send_input(mapped_action)
                # Advance 3 frames
                self.pyboy.tick()
                self.pyboy.tick()
                self.pyboy.tick()
                self.frame_count += 3
                # Release the button
                self.release_button(mapped_action)  # You can use your existing release_button function
        finally:
            if self.action_lock.locked():
                self.action_lock.release() # Release the lock when done.

    def release_button(self, mapped_action):
        release_map = {
            WindowEvent.PRESS_ARROW_UP: WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.PRESS_ARROW_DOWN: WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT: WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.PRESS_BUTTON_A: WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B: WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.PRESS_BUTTON_SELECT: WindowEvent.RELEASE_BUTTON_SELECT,
            WindowEvent.PRESS_BUTTON_START: WindowEvent.RELEASE_BUTTON_START,
        }
        self.pyboy.send_input(release_map.get(mapped_action, None))

    def close(self):
        self.pyboy.stop()
