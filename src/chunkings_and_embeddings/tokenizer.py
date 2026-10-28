from transformers import AutoTokenizer

# Instanciando o tokenizer compatível com o modelo de embeddings
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')

tokens = tokenizer.encode("Exemplo de texto para tokenização", add_special_tokens=True)
print(tokens)