from configs.globals import task_queue, done_queue
import asyncio


async def generate_joke(name):
    print(f"Worker-{name} started")
    import pickle

    with open("../../data/article_texts.txt", 'rb') as f:
        texts = pickle.load(f, encoding="UTF-8")
    with open("../../data/english_anecs_list.pickle", "rb") as f:
        english_anecs_list = pickle.load(f, encoding="UTF-8")
    with open("../../data/ids_to_labels.pickle", "rb") as f:
        ids_to_labels = pickle.load(f, encoding="utf-8")
    with open("../../data/labels_to_ids.pickle", "rb") as f:
        labels_to_ids = pickle.load(f, encoding="utf-8")
    with open("../../data/unique_labels.pickle", "rb") as f:
        unique_labels = pickle.load(f, encoding="UTF-8")
    with open("../../data/translated_anecs.txt", "r") as f:
            translated_anecs = f.read().replace("<unk> ", "").replace("♪ ", "").split("\n")
    with open("../../data/translated_anecs_prepared.pickle", "rb") as f:
        translated_anecs_prepared = pickle.load(f)
    with open("../../data/english_anecs_prepared.pickle", "rb") as f:
        english_anecs_prepared = pickle.load(f)
    with open("../../data/glove_vectors.pickle", "rb") as f:
        glove_vectors = pickle.load(f)
    print("Downloaded needed data")
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained('dslim/bert-base-NER')
    print("Downloaded tokenizer")
    def tokenize(data: str):
        inputs = tokenizer(data, return_tensors="pt", truncation=True, padding=True)
        return inputs

    def ids_to_tokens(text_input):
        return tokenizer.convert_ids_to_tokens(text_input)

    from transformers import BertForTokenClassification
    import torch

    class BertModel(torch.nn.Module):
        def __init__(self):
            super(BertModel, self).__init__()
            self.bert = BertForTokenClassification.from_pretrained('dslim/bert-base-NER',
                                                                   num_labels=len(unique_labels),
                                                                   ignore_mismatched_sizes=True)

        def forward(self, input_ids, label=None):
            output = self.bert(labels=label, input_ids=input_ids, return_dict=False)
            return output

    model = BertModel()
    try:
        model.load_state_dict(torch.load('models/bert_trainedNEREnglish', map_location=torch.device('cpu')))
    except Exception as e:
        print("Alarm", e)
    print("Downloaded pretrained model")
    import numpy as np

    def get_ners(text: str) -> list:
        output = []
        B = np.asarray([tokenizer(text.replace("-", ""))["input_ids"]]).reshape(1, 1, -1)
        logits = model(torch.as_tensor(np.array(B))[0])[0]
        for j in range(logits.shape[0]):
            logits_clean = logits[j].argmax(dim=1)
            tokenized_sentence = ids_to_tokens(tokenizer(text.replace("-", ""))["input_ids"])
            # for i in range(len(logits_clean)):
            #    print(tokenized_sentence[i], ids_to_labels[logits_clean[i].item()])
            # print([ids_to_labels[x.item()] for x in logits_clean])
            i = 1
            for elem in logits_clean[1:-1]:
                if i > 1 and (tokenized_sentence[i][:2] == "##" or ids_to_labels[elem.item()][0] == "I"):
                    if tokenized_sentence[i][:2] == "##":
                        output[-1]["word"] += tokenized_sentence[i][2:]
                    else:
                        output[-1]["word"] += tokenized_sentence[i]
                else:
                    output.append({"word": tokenized_sentence[i], "entity": ids_to_labels[elem.item()]})
                i += 1
        return output
    print("Downloaded glove-wiki-gigaword-300 ...")
    def get_embeddings(list_of_tags: list):
        emeddings = []
        for tag in list_of_tags:
            try:
                embed = glove_vectors[tag["word"]]
                emeddings.append({'entity': tag["entity"], 'word': tag["word"], "embedding": embed})
            except:
                emeddings.append(
                    {'entity': tag["entity"], 'word': tag["word"], "embedding": glove_vectors["base"]})
        return emeddings

    def get_non_o(ner_words):
        a = []
        for x in ner_words:
            if x["entity"] != "O":
                a.append(x)
        return a

    from numpy.linalg import norm
    from collections import Counter

    def count_cos_embeddings(text_embeddings, anec_embeddings) -> (float, dict):
        suitable_pairs = []
        anec_unique_tags_counter = Counter()
        embedding_cosine_sum = 0
        for embedding in text_embeddings:
            cosines = []
            simmilarity_tags = {}
            for embed in anec_embeddings:
                anec_unique_tags_counter[embed["entity"]] += 1
                if embed["entity"] == embedding["entity"]:
                    v1 = embedding["embedding"]
                    v2 = embed["embedding"]
                    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cosines.append((cos, embed["word"], embed["entity"]))
            simmilarity_tags[embedding["word"]] = cosines
            suitable_pairs.append(simmilarity_tags)
        top_similarity = {}
        for word_dict in suitable_pairs:
            word = list(word_dict.keys())[0]
            top = (-10, "")
            key = None
            for sim in word_dict[word]:
                if top[0] < sim[0]:
                    top = (sim[0], word)
                    key = sim[1]
            if key:
                top_similarity[key] = top
        for key in top_similarity.keys():
            embedding_cosine_sum += top_similarity[key][0]
        embedding_cosine_sum /= len(top_similarity.keys()) if len(top_similarity.keys()) > 5 else 5
        return embedding_cosine_sum, top_similarity

    def get_resulted_text(text: str, swear_flag):
        best_anec = ""
        best_cos = -10
        best_simmilarity = None
        if swear_flag:
            resulted_anec_list = english_anecs_prepared + translated_anecs_prepared
        else:
            resulted_anec_list = english_anecs_prepared
        filtred_text = get_non_o(get_ners(text))
        text_embeddings = get_embeddings(filtred_text)
        for anec_embed, anec in resulted_anec_list:
            try:
                new_cosine, new_simmilarity = count_cos_embeddings(text_embeddings, anec_embed)
                if new_cosine > best_cos:
                    best_cos = new_cosine
                    best_filtered = anec_embed
                    best_text_ner = filtred_text
                    best_anec = anec
                    best_simmilarity = new_simmilarity
            except Exception as e:
                pass

        words = best_anec.split()
        key_words = best_simmilarity.keys()
        for i in range(len(words)):
            if words[i] in key_words:
                words[i] = best_simmilarity[words[i]][1]
        resulted_text = " ".join(words)
        return resulted_text

    try:
        while True:
            task_id, text, swear_flag = await task_queue.get()
            done_queue.put_nowait((task_id, get_resulted_text(text, swear_flag)))
            task_queue.task_done()
    except Exception as e:
        print(e)
        print(f"Worker-{name} stopped")


async def start_workers(N_of_workers:int = 1):
    tasks = []
    ids = {}
    for i in range(N_of_workers):
        task = asyncio.create_task(generate_joke(f'worker-{i}'))
        tasks.append(task)
    return tasks

