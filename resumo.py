import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from heapq import nlargest
import string
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

bot = ChatBot('Bot')
chatbot = ChatBot(
    'SENAC2',
    logic_adapters=[
        'chatterbot.logic.BestMatch',
        'chatterbot.logic.TimeLogicAdapter']
)

trainer = ListTrainer(bot)
trainer.train([
	'oi',
    'Seja bem vindo, sou um bot que resume textos, voce deseja resumir algum texto? ',
    'obrigado',
    'fico feliz em ajudar, voce deseja resumir mais algum texto?',

])

def preprocess_text_pt(text):
    tokens = word_tokenize(text.lower(), language='portuguese')
    stop_words = set(stopwords.words('portuguese'))
    words = [word for word in tokens if word.isalnum() and word not in stop_words and word not in string.punctuation]
    return " ".join(words)


def generate_summary(text, num_sentences):
    sentences = sent_tokenize(text, language='portuguese')

    preprocessed_text = preprocess_text_pt(text)

    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('portuguese'))
    tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_text])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    word_scores = {}

    for word, score in zip(feature_names, tfidf_matrix.toarray()[0]):
        word_scores[word] = score

    sentence_scores = {}

    for i, sentence in enumerate(sentences):
        sentence_words = word_tokenize(sentence.lower(), language='portuguese')
        score = sum(word_scores[word] for word in sentence_words if word in word_scores)
        sentence_scores[i] = score / len(sentence_words) if len(sentence_words) > 0 else 0

    selected_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(sentences[i] for i in sorted(selected_sentences))

    return summary

while True:
    request = input('you: ')
    if request.lower() == 'sim':
        print('ok')
        request = input("Digite o texto: ")
        summary = generate_summary(request, 2)
        print(summary)
    elif request.lower() in ['nao', 'não']:
        print('ok, tchau')
        break
    else:
        response = bot.get_response(request)
        print('bot: ', response)