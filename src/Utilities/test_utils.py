from Utilities.SentenceTokenizer import tokenizeSentence
from Utilities.SentenceTokenizer import span_tokenizer

sentence = "This is a sentence about Mt. St. Helens, though I see how there could be some confusion: I'm a duck."
tokenized_sentence = tokenizeSentence(sentence)
print(tokenized_sentence)

