


# conda activate C:\AI\text-generation-webui\installer_files\env
# cd C:\Users\linoa\Documents\Code\LMM_Pokemon
# python -m llava.serve.controller --host 0.0.0.0 --port 10000
# 
# python server.py --api --model models/wojtab_llava-13b-v0-4bit-128g --multimodal-pipeline llava-13b


# from pyboy import PyBoy
# pyboy = PyBoy('ROMs/Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb')
# while not pyboy.tick():
#     pass
# pyboy.stop()

from env import Env

def main(rom_path):
    env = Env(rom_path)
    
    try:
        # Run for a number of steps as a demo
        while True:
            if env.step():  # Modify the step() method to return True when window is closed
                print("PyBoy window closed.")
                break
    except KeyboardInterrupt:
        print("Exiting environment...")
    finally:
        
      
        # Save to file
        
        env.close()



if __name__ == "__main__":
   main(r'C:\Users\linoa\Documents\Code\LMM_Pokemon\ROMs\Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb')