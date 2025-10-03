from sentence_transformers import SentenceTransformer

model = SentenceTransformer('/workspace/juhao/adaptive_agent/GDesigner/data/all-MiniLM-L6-v2')

def get_sentence_embedding(sentence):
    embeddings = model.encode(sentence)
    return embeddings
