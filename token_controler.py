from transformers import GPT2Tokenizer

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def limit_tokens(text, max_tokens=250):
    """
    Limit the number of tokens in the text to max_tokens.
    """
    tokens = tokenizer.encode(text)
    limited_tokens = tokens[:max_tokens]
    # Decode back to text
    limited_text = tokenizer.decode(limited_tokens, clean_up_tokenization_spaces=True)
    return limited_text