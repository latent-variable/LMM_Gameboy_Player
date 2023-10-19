

def get_gameplay_prompt():
    valid_options = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'SELECT', 'START']
    valid_options_str = ', '.join(valid_options)
    prompt = f"You are playing a game, what should the next move be? {valid_options_str}\n"

    return prompt