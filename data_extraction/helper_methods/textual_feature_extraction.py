import os
import re
import math
import nltk
import torch
import fasttext

import pandas as pd
import numpy as np
import urllib.request
from tqdm import tqdm

from collections import Counter

from nltk.tokenize import word_tokenize , sent_tokenize
from nltk.corpus import stopwords,wordnet
from nltk.util import ngrams 

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util


from transformers import AutoModel, AutoTokenizer,pipeline
from transformers import AutoModel, AutoTokenizer, AutoConfig

    
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4') 


class textual_feature_extractors:
    def __init__(self):

        #  THIS WAS ALL AI AND IT WAS ANNOYING AS FUUUUCK - sorry for my colourfull language
        # Your perfect path configuration
        venv_nltk_data = '/home/marcos-vargas/Documents/PROJECT_COMMAXX/OFFICIAL_ANALYSIS/venv/nltk_data'
        os.makedirs(venv_nltk_data, exist_ok=True)

        # Insert at position 0 to make it the absolute highest priority path
        nltk.data.path.insert(0, venv_nltk_data)

        # Map each package to the path nltk.data.find() uses to locate it
        required_packages = {
            'punkt': 'tokenizers/punkt',
            'punkt_tab': 'tokenizers/punkt_tab',
            'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
            'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng',
        }

        # Only download a package if it isn't already present (avoids re-downloading every run)
        for package, find_path in required_packages.items():
            try:
                nltk.data.find(find_path)
            except LookupError:
                nltk.download(package, download_dir=venv_nltk_data)

        # Resolve the model path relative to this file (not the current working directory)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fasttext_model_path = os.path.join(project_root, 'lid.176.ftz')

        # Download the language-ID model if it isn't already present
        if not os.path.exists(fasttext_model_path):
            url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz'
            urllib.request.urlretrieve(url, fasttext_model_path)

        self.model_fast_text = fasttext.load_model(fasttext_model_path)

        self.lang_map = {
            'en': 'eng',
            'fr': 'fra',
            'es': 'spa',
            'de': 'deu',
            'nl': 'nld', 
            'it': 'ita',
            'pt': 'por'
            }

        self.lang_map_nltk = {
            'en': 'english',
            'fr': 'french',
            'es': 'spanish',
            'de': 'german',
            'it': 'italian',
            'pt': 'portuguese',
            'nl': 'dutch'
        }

        self.writing_labels = ["descriptive", "persuasive", "direct", "technical" ]
        self.formality_labels = ["formal" , "casual" , "informal"]
        self.tone_labels = ["enthusiastic", "clinical", "urgent" ,"confident" , "idealistic" , "realistic" ]
        self.quality_labels = ["benefit-led", "generic", "scannable", "keyword-stuffed"]


        self.all_labels = self.writing_labels + self.formality_labels + self.tone_labels + self.quality_labels



        text_model_name = "nomic-ai/nomic-embed-text-v1.5"
        self.text_model = SentenceTransformer(text_model_name, trust_remote_code=True)


        model_name = "nomic-ai/nomic-embed-text-v1.5"

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_hidden_states = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        self.model.eval()




        self.classifier = pipeline(
        "zero-shot-classification", 
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        device=0,                       
        dtype=torch.float16       # Doubles speed on modern GPUs (transformers renamed torch_dtype -> dtype)
        )



    def extract_POS_tags(self, text):
        tokens = word_tokenize(text)
        tagged_tokens = nltk.pos_tag(tokens)
        return tagged_tokens

    def extract_unique_word(self, text):

        tagged_tokens = self.extract_POS_tags(text)

        target_POS = {
            'NN', 'NNS', 'NNP', 'NNPS',  # Nouns
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', # Verbs
            'JJ', 'JJR', 'JJS',          # Adjectives
            'RB', 'RBR', 'RBS'           # Adverbs
        }
        unique_tokens = []
        for tup in tagged_tokens:
            word , tag  = tup 
            if tag in target_POS:
                unique_tokens.append(word)
        return unique_tokens

    def extract_lexical_denisty(self, text):
        tokens = word_tokenize(text)
        unique_tokens = self.extract_unique_word(text)

        return float(len(unique_tokens))/float(len(tokens))
    
    def extract_punctuation_entropy(self , text):
        punctuation_list = re.findall(r'[^\w\s]', text)
        punctuation_counts = Counter(punctuation_list)

        total = punctuation_counts.total() 

        punctuaion_entropy = 0 
        for punct , count in punctuation_counts.items():
            prob= count/total
            punctuaion_entropy += (prob)* math.log(prob, 2)

        return -punctuaion_entropy
    

    def detect_language(self, text):
        text = text.replace('\n', ' ')
        # Call the underlying model directly instead of self.model_fast_text.predict().
        # fasttext's predict() wrapper calls np.array(probs, copy=False), which NumPy 2.x
        # rejects. Args are (text, k, threshold, on_unicode_error); it returns a list of
        # (probability, label) tuples, so we read [0][1] for the top label.
        predictions = self.model_fast_text.f.predict(text, 1, 0.0, 'strict')
        # fastText returns [] for very short / low-signal text (e.g. 1-2 words, blanks).
        # Fall back to English so callers don't crash on an empty result.
        if not predictions:
            return 'en'
        lang_code = predictions[0][1].replace('__label__', '')
        return lang_code 
    

    def extract_imperatives(self, text):
        sentences = sent_tokenize(text)
        imperative_count = 0
        imperative_words = []
        
        for sent in sentences:
            words = word_tokenize(sent)
            tags = nltk.pos_tag(words)
            
            for i, (word, tag) in enumerate(tags):
                if tag == 'VB':
                    if i == 0:
                        imperative_count += 1
                        imperative_words.append(word)
                    
                    elif tags[i-1][1] in [',', 'CC']:
                        imperative_count += 1
                        imperative_words.append(word)

        return imperative_count, imperative_words


   

    def extract_stop_words(self, text):
        text = re.sub(r'[^\w\s]', ' ', text)
        lang = self.detect_language(text)

        nltk_lang = self.lang_map_nltk.get(lang, 'english')
        stopword_set = set(stopwords.words(nltk_lang))

        tokens = word_tokenize(text)

        found_stopwords = []

        for tk in tokens:
            word = tk.lower()
            if word in stopword_set:
                found_stopwords.append(word)

        return found_stopwords
    

    #  Extracting amount of times synonyms of the target keyword have been found 

    def get_synonymes(self, key_word , lang_code):
        synonyms = {key_word}
        wordnet_lang = self.lang_map.get(lang_code, 'eng')

        for syn in wordnet.synsets(key_word, lang=wordnet_lang):
            for lemma in syn.lemmas(lang=wordnet_lang):
                synonyms.add(lemma.name().replace('_' , ' '))
        
        return synonyms


    #  extracts how many times the keyword and its synonymes have been found in the text
    def extract_keyword_count(self, key_word, text):
        text  = re.sub(r'[^\w\s]', ' ', text)

        lang  = self.detect_language(text)
        synonyms = list(self.get_synonymes(key_word, lang))

        tokens  = word_tokenize(text)
        all_tokens = tokens.copy()

        bigrams = [' '.join(gram) for gram in ngrams(tokens, 2)]
        all_tokens.extend(bigrams)
        
        # trigrams = [' '.join(gram) for gram in ngrams(tokens, 3)]
        # all_tokens.extend(trigrams)

        embeddings =  self.text_model.encode( all_tokens, show_progress_bar=False, convert_to_tensor=True )
        target_embd = self.text_model.encode(synonyms, show_progress_bar=False, convert_to_tensor=True )



        THRESHOLD = 0.57# Note: You want > 0.7 for synonyms.
        candidates = [] 
        
        #  the end vector is a weighted sum of the synonymes with respect to the key word 
        cosine_sim_syn_to_keyWord = util.cos_sim(target_embd[0] , target_embd)
        best_embd = []



        cosine_sim_syn_to_keyWord = cosine_sim_syn_to_keyWord.flatten().tolist()
        for i , entry in enumerate(target_embd):
            if cosine_sim_syn_to_keyWord[i] >0.55:
                best_embd.append(entry)

        best_embd = torch.stack(best_embd)
        cosine_scores = util.cos_sim(best_embd, embeddings)
        max_scores = cosine_scores.max(dim=0).values

        for i, token in enumerate(all_tokens):
            if max_scores[i].item() > THRESHOLD:
                candidates.append(token)
                

        return candidates 
    

    def extract_length(self, text):
        tokens = word_tokenize(text)
        return len(tokens)
    

    def extract_optimal_embedding(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        with torch.no_grad():
            x = self.model.embeddings(
                input_ids=inputs["input_ids"]
            )

            x = self.model.emb_ln(x)

            hidden_states = []

            for layer in self.modelmodel.encoder.layers:
                x = layer(hidden_states=x, hidden_states2=x)[0]
                hidden_states.append(x)

        layers = hidden_states[6:8]

        stacked = torch.stack(layers)
        style_emb = stacked.mean(dim=0).mean(dim=1)

        return style_emb.cpu().numpy()
    
    # STYLISTIC DISTRIBUTION OF TEXT BASED ON PROJECT 1,2

    
    def format(self, results):
        row2 = results['scores']
        row = {}
        for i , key in enumerate(results['labels']):
            row[key] = row2[i]

        return row


    def stylistic_distribution_extraction(self, index  , text):

        # Guard against None / NaN / empty cells coming from the paragraph DataFrame.
        if isinstance(text, str) and text.strip():
            writing_results = self.classifier(text , self.writing_labels)
            formality_results = self.classifier(text, self.formality_labels)
            tone_results = self.classifier(text, self.tone_labels)
            quality_results = self.classifier(text,self.quality_labels)


            writing_result_row = self.format(writing_results)
            formality_results_row = self.format(formality_results)
            tone_results_row= self.format(tone_results)
            quality_results_row = self.format(quality_results)


            row = [index , text, writing_result_row, formality_results_row, tone_results_row, quality_results_row]
            return row


    # Names of every dimension returned by extract_stylistic_vector, in order.
    # Use this to label columns / interpret the vector and to size the empty-text case.
    STYLISTIC_FEATURES = [
        "lexical_density", "stopword_ratio", "type_token_ratio", "hapax_ratio",
        "avg_sentence_length", "std_sentence_length", "max_sentence_length",
        "avg_word_length", "long_word_ratio", "uppercase_ratio", "capitalized_ratio",
        "digit_token_ratio",
        "punctuation_entropy", "punctuation_ratio", "exclamation_ratio",
        "question_ratio", "comma_ratio", "imperative_ratio",
        "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio", "pronoun_ratio",
        "determiner_ratio", "conjunction_ratio", "numeral_ratio",
        "comparative_superlative_ratio",
    ]

    def extract_stylistic_vector(self, text):
        # Empty / whitespace-only text -> zero vector of the right length (no crashes).
        if not text or not str(text).strip():
            return np.zeros(len(self.STYLISTIC_FEATURES))

        tokens = word_tokenize(text)
        sentences = sent_tokenize(text)

        # max(..., 1) everywhere so a one-token / one-sentence paragraph can't divide by zero.
        num_tokens = max(len(tokens), 1)
        num_sentences = max(len(sentences), 1)
        num_chars = max(len(text), 1)

        # ---- sentence stats ----
        sentence_lengths = [len(word_tokenize(s)) for s in sentences] or [0]
        avg_sentence_length = float(np.mean(sentence_lengths))
        std_sentence_length = float(np.std(sentence_lengths))
        max_sentence_length = float(np.max(sentence_lengths))

        # ---- POS ratios ----
        tags = nltk.pos_tag(tokens)
        pos_counts = Counter(tag for _, tag in tags)
        def pos_ratio(tagset):
            return sum(pos_counts[t] for t in tagset) / num_tokens

        noun_ratio = pos_ratio(['NN', 'NNS', 'NNP', 'NNPS'])
        verb_ratio = pos_ratio(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
        adj_ratio  = pos_ratio(['JJ', 'JJR', 'JJS'])
        adv_ratio  = pos_ratio(['RB', 'RBR', 'RBS'])
        pronoun_ratio = pos_ratio(['PRP', 'PRP$'])
        determiner_ratio = pos_ratio(['DT'])
        conjunction_ratio = pos_ratio(['CC'])
        numeral_ratio = pos_ratio(['CD'])
        # Comparatives / superlatives are a strong marketing-language signal.
        comparative_superlative_ratio = pos_ratio(['JJR', 'JJS', 'RBR', 'RBS'])

        # ---- lexical / vocabulary richness ----
        lexical_density = self.extract_lexical_denisty(text)
        stopword_ratio = len(self.extract_stop_words(text)) / num_tokens
        type_token_ratio = len(set(tokens)) / num_tokens
        freq = Counter(tokens)
        hapax_ratio = sum(1 for _, c in freq.items() if c == 1) / len(freq)

        # ---- word shape ----
        alpha = [t for t in tokens if t.isalpha()]
        avg_word_length = float(np.mean([len(t) for t in alpha])) if alpha else 0.0
        long_word_ratio = sum(1 for t in alpha if len(t) > 6) / num_tokens
        uppercase_ratio = sum(1 for t in alpha if len(t) > 1 and t.isupper()) / num_tokens
        capitalized_ratio = sum(1 for t in alpha if t.istitle()) / num_tokens
        digit_token_ratio = sum(1 for t in tokens if any(ch.isdigit() for ch in t)) / num_tokens

        # ---- punctuation ----
        punctuation_entropy = self.extract_punctuation_entropy(text)
        punctuation_ratio = len(re.findall(r'[^\w\s]', text)) / num_chars
        exclamation_ratio = text.count('!') / num_sentences
        question_ratio = text.count('?') / num_sentences
        comma_ratio = text.count(',') / num_sentences

        # ---- pragmatics ----
        imperative_count, _ = self.extract_imperatives(text)
        imperative_ratio = imperative_count / num_sentences

        # Order MUST match STYLISTIC_FEATURES above.
        return np.array([
            lexical_density,
            stopword_ratio,
            type_token_ratio,
            hapax_ratio,
            avg_sentence_length,
            std_sentence_length,
            max_sentence_length,
            avg_word_length,
            long_word_ratio,
            uppercase_ratio,
            capitalized_ratio,
            digit_token_ratio,
            punctuation_entropy,
            punctuation_ratio,
            exclamation_ratio,
            question_ratio,
            comma_ratio,
            imperative_ratio,
            noun_ratio,
            verb_ratio,
            adj_ratio,
            adv_ratio,
            pronoun_ratio,
            determiner_ratio,
            conjunction_ratio,
            numeral_ratio,
            comparative_superlative_ratio,
        ])
                
    