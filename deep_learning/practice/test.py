import spacy
import en_core_web_sm

nlp = en_core_web_sm.load()
doc = nlp(u"This is a sentence.")

print(doc)