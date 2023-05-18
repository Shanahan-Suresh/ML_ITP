import re

def custom_tokenizer(s):
    return re.findall(r'\w+|[^\w\s]', s)

