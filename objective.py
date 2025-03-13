import re
import nltk
import numpy as np
from nltk.corpus import wordnet as wn

class ObjectiveTest:

    def __init__(self, filepath, noOfQues):
        self.summary = filepath
        self.noOfQues = noOfQues

    def get_trivial_sentences(self):
        sentences = nltk.sent_tokenize(self.summary)
        trivial_sentences = list()
        for sent in sentences:
            trivial = self.identify_trivial_sentences(sent)
            if trivial:
                trivial_sentences.append(trivial)
        return trivial_sentences

    def identify_trivial_sentences(self, sentence):
        # Correctly tokenize the sentence into words before POS tagging
        tokens = nltk.word_tokenize(sentence)  # Tokenize sentence to a list of words
        tags = nltk.pos_tag(tokens)  # POS tagging on list of words
        
        if tags[0][1] == "RB" or len(tokens) < 4:  # Check token length after tokenization
            return None
        
        noun_phrases = list()
        grammar = r"""
            CHUNK: {<NN>+<IN|DT>*<NN>+}
                   {<NN>+<IN|DT>*<NNP>+}
                   {<NNP>+<NNS>*}
            """
        chunker = nltk.RegexpParser(grammar)
        tree = chunker.parse(tags)  # Parse POS tags into chunks

        for subtree in tree.subtrees():
            if subtree.label() == "CHUNK":
                temp = " ".join(word for word, _ in subtree)
                noun_phrases.append(temp)
        
        replace_nouns = []
        for word, _ in tags:
            for phrase in noun_phrases:
                if phrase[0] == '\'':
                    break
                if word in phrase:
                    [replace_nouns.append(phrase_word) for phrase_word in phrase.split()[-2:]]
                    break
            if len(replace_nouns) == 0:
                replace_nouns.append(word)
            break
        
        if len(replace_nouns) == 0:
            return None
        
        val = min(len(i) for i in replace_nouns)  # Get the minimum length of the replaceable nouns
        
        trivial = {
            "Answer": " ".join(replace_nouns),
            "Key": val
        }

        if len(replace_nouns) == 1:
            trivial["Similar"] = self.answer_options(replace_nouns[0])
        else:
            trivial["Similar"] = []
        
        replace_phrase = " ".join(replace_nouns)
        blanks_phrase = ("__________ " * len(replace_nouns)).strip()
        expression = re.compile(re.escape(replace_phrase), re.IGNORECASE)
        sentence = expression.sub(blanks_phrase, sentence, count=1)
        trivial["Question"] = sentence
        return trivial

    @staticmethod
    def answer_options(word):
        synsets = wn.synsets(word, pos="n")

        if not synsets:
            return []
        else:
            synset = synsets[0]
        
        hypernyms = synset.hypernyms()
        if not hypernyms:
            return []
        
        hypernym = hypernyms[0]
        hyponyms = hypernym.hyponyms()
        similar_words = []
        for hyponym in hyponyms:
            similar_word = hyponym.lemmas()[0].name().replace("_", " ")
            if similar_word != word:
                similar_words.append(similar_word)
            if len(similar_words) == 8:
                break
        return similar_words

    def generate_test(self):
        trivial_pair = self.get_trivial_sentences()
        question_answer = list()
        for que_ans_dict in trivial_pair:
            if que_ans_dict["Key"] > int(self.noOfQues):
                question_answer.append(que_ans_dict)
        
        question = list()
        answer = list()
        while len(question) < int(self.noOfQues):
            rand_num = np.random.randint(0, len(question_answer))
            if question_answer[rand_num]["Question"] not in question:
                question.append(question_answer[rand_num]["Question"])
                answer.append(question_answer[rand_num]["Answer"])
        return question, answer
