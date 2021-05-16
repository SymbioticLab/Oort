import tensorflow as tf
import tensorflow_federated as tff
import pickle

"""

Download token and tag dict from tff

"""

def create_tag_vocab(vocab_size):
  """Creates vocab from `vocab_size` most common tags in Stackoverflow."""
  tag_dict = tff.simulation.datasets.stackoverflow.load_tag_counts()
  return list(tag_dict.keys())
def create_token_vocab(vocab_size):
  """Creates vocab from `vocab_size` most common words in Stackoverflow."""
  vocab_dict = tff.simulation.datasets.stackoverflow.load_word_counts()
  return list(vocab_dict.keys())


vocab_tokens = create_token_vocab(10000)
vocab_tags = create_tag_vocab(500)

token_file = "vocab_tokens"
with open(token_file, 'wb') as f:
  pickle.dump(vocab_tokens, f)


tag_file = "vocab_tags.txt"
with open(tag_file, 'wb') as f
  pickle.dump(vocab_tags, f)


