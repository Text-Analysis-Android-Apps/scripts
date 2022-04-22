from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TFT5ForConditionalGeneration, TFPreTrainedModel, T5Tokenizer, T5ForConditionalGeneration

#model_name = "Vamsi/T5_Paraphrase_Paws"
model_name = './T5_Paraphrase_Paws'

#model = AutoModelForSeq2SeqLM.from_pretrained(model_name, from_pt=True)
model = TFT5ForConditionalGeneration.from_pretrained(model_name, local_files_only=True)
tokenizer = T5Tokenizer.from_pretrained(model_name, local_files_only=True)

#sentence = "This is something which i cannot understand at all"
sentence = "Sergio like going to the library"

text =  "paraphrase: " + sentence + " </s>"


max_len = 256

encoding = tokenizer.encode_plus(text, padding='max_length', return_tensors="tf")
input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]


outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    do_sample=True,
    max_length=256,
    top_k=220,
    top_p=1,
    early_stopping=True,
    num_return_sequences=5
)

print
print("Outputs: "+str(outputs.shape.as_list()))

print ("\nOriginal Question ::")
print (sentence)
print ("\n")
print ("Paraphrased Questions :: ")
final_outputs =[]
for output in outputs:
    sent = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    if sent.lower() != sentence.lower() and sent not in final_outputs:
        final_outputs.append(sent)

for i, final_output in enumerate(final_outputs):
    print("{}: {}".format(i, final_output))
