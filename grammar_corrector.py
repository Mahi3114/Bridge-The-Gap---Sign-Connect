# # grammar_corrector.py - Module for grammar correction and paraphrasing using Hugging Face Transformers
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # --------- Load Grammar Correction Model ---------
# gc_tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
# gc_model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")

# # --------- Load Paraphraser Model ---------
# para_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
# para_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")

# def polish_sentence(text):
#     try:
#         # --- Step 1: Grammar Correction ---
#         gc_input = "gec: " + text
#         gc_ids = gc_tokenizer.encode(gc_input, return_tensors="pt", truncation=True)
#         gc_outputs = gc_model.generate(gc_ids, max_length=128, num_beams=5, early_stopping=True)
#         corrected = gc_tokenizer.decode(gc_outputs[0], skip_special_tokens=True)

#         # --- Step 2: Paraphrasing ---
#         para_input = "paraphrase: " + corrected + " </s>"
#         enc = para_tokenizer.encode_plus(para_input, return_tensors="pt", padding="longest")
#         para_outputs = para_model.generate(
#             input_ids=enc["input_ids"],
#             attention_mask=enc["attention_mask"],
#             max_length=128,
#             num_beams=5,
#             num_return_sequences=1,
#             early_stopping=True
#         )
#         polished = para_tokenizer.decode(para_outputs[0], skip_special_tokens=True)

#         return polished

#     except Exception as e:
#         print("⚠ Error in pipeline:", e)
#         return text


# # ---------------- Example ----------------
# raw_text = "he go to school yesterday and play football"
# print("Input :", raw_text)
# print("Output:", polish_sentence(raw_text))



# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# gc_tokenizer = None
# gc_model = None
# para_tokenizer = None
# para_model = None

# def load_models():
#     global gc_tokenizer, gc_model, para_tokenizer, para_model

#     if gc_tokenizer is None:
#         print("Loading Grammar Model...")
#         gc_tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
#         gc_model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")

#     if para_tokenizer is None:
#         print("Loading Paraphrase Model...")
#         para_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
#         para_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")


# def polish_sentence(text):
#     try:
#         load_models()  # 🔥 LOAD ONLY WHEN NEEDED

#         # Grammar correction
#         gc_input = "gec: " + text
#         gc_ids = gc_tokenizer.encode(gc_input, return_tensors="pt", truncation=True)
#         gc_outputs = gc_model.generate(gc_ids, max_length=128, num_beams=5)
#         corrected = gc_tokenizer.decode(gc_outputs[0], skip_special_tokens=True)

#         # Paraphrase
#         para_input = "paraphrase: " + corrected + " </s>"
#         enc = para_tokenizer.encode_plus(para_input, return_tensors="pt", padding="longest")
#         para_outputs = para_model.generate(
#             input_ids=enc["input_ids"],
#             attention_mask=enc["attention_mask"],
#             max_length=128,
#             num_beams=5
#         )
#         polished = para_tokenizer.decode(para_outputs[0], skip_special_tokens=True)

#         return polished

#     except Exception as e:
#         print("Grammar error:", e)
#         return text


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

gc_tokenizer = None
gc_model = None
para_tokenizer = None
para_model = None

def load_models():
    global gc_tokenizer, gc_model, para_tokenizer, para_model

    if gc_tokenizer is None:
        print("Loading Grammar Model...")
        gc_tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
        gc_model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")

    if para_tokenizer is None:
        print("Loading Paraphrase Model...")
        para_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        para_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")


def polish_sentence(text):
    try:
        load_models()  # 🔥 LOAD ONLY WHEN NEEDED

        # Grammar correction
        gc_input = "gec: " + text
        gc_ids = gc_tokenizer.encode(gc_input, return_tensors="pt", truncation=True)
        gc_outputs = gc_model.generate(gc_ids, max_length=128, num_beams=5)
        corrected = gc_tokenizer.decode(gc_outputs[0], skip_special_tokens=True)

        # Paraphrase
        para_input = "paraphrase: " + corrected + " </s>"
        enc = para_tokenizer.encode_plus(para_input, return_tensors="pt", padding="longest")
        para_outputs = para_model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_length=128,
            num_beams=5
        )
        polished = para_tokenizer.decode(para_outputs[0], skip_special_tokens=True)

        return polished

    except Exception as e:
        print("Grammar error:", e)
        return text