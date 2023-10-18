

def get_gameplay_prompt():
    valid_options = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'SELECT', 'START']
    prompt = f"Give a short description of the game screen, then provide me with instructs on what to do next to advance in the game."

    return prompt