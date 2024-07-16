from emoji import demojize, replace_emoji
from nltk.corpus import stopwords
import re

def replace_username(text):
  user_regex = '(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)'
  return re.sub(user_regex, '@USER', text)

def replace_hashtag(text):
  hashtag_regex = '\B#\w*[a-zA-Z]+\w*'
  return re.sub(
      hashtag_regex,
      lambda x: 'hashtag '+ x.group().replace('_','')[1:],
      text
    )

def replace_emoji_with_token(text):
  return replace_emoji(text, replace=lambda x, y: 'emoji '+demojize(x)[1:-1]+' emoji')

def clean_text(text):
  sw = stopwords.words('english')
  text = text.lower()

  text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text) # replacing everything with space except (a-z, A-Z, ".", "?", "!", ","
  text = re.sub(r"http\S+", "",text) #Removing URLs
  html=re.compile(r'<.*?>')
  text = html.sub(r'',text) #Removing html tags

  punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
  for p in punctuations:
    text = text.replace(p,'') #Removing punctuations

  text = [word.lower() for word in text.split() if word.lower() not in sw]
  text = " ".join(text) #removing stopwords

  return text

def preprocess_text(text):
  return clean_text(
    replace_emoji_with_token(
      replace_hashtag(
        replace_username(text)
      )
    )
  )