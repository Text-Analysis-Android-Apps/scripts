import json
from transformers import T5Tokenizer
from google.colab import files

tokenizer = T5Tokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")

vocab = tokenizer.get_vocab()

with open("vocab.json", "w") as write_file:
    json.dump(vocab, write_file, indent=4)

with open("vocab.txt", "w") as write_file:
    for word in vocab:
      write_file.write(word+"\n")

files.download("vocab.json")
files.download("vocab.txt")
