import pickle
import time
from pathlib import Path

import pandas as pd
import torch
import typer
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
from nltk import tokenize

print("hello")

def cls_pooling(model_output, attention_mask):
    return model_output[0][:, 0]


def max_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    return torch.max(token_embeddings, 1)[0]


def main(
        input_model: str = typer.Option(default="dmis-lab/biobert-base-cased-v1.2",  # , paraphrase-MiniLM-L6-v2
                                        help="Path to the input model"),
        pooling: str = typer.Option(default="mean", help="Options of mean, max or cls"),
        input_data_file: Path = typer.Option(default="data/SUBJECT_ID_to_NOTES_1a_2000.csv",
                                             help="Path to the input model"),
        type_cluster: str = typer.Option(default="kmeans", help="Options of kmeans, fast, topic")
):
    ###################  Get Embeddings with BERT ###################

    corpus_df = pd.read_csv(input_data_file)[:200]
    corpus = corpus_df["TEXT"].to_list()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sentences = []
    for p in corpus: 
        sentence = tokenize.sent_tokenize(p)
        sentences.extend(sentence)

    if pooling == "cls":
        tokenizer = AutoTokenizer.from_pretrained(input_model)
        model = AutoModel.from_pretrained(input_model)
        sentence_embeddings = []
        batch_size = 20
        for i in range(len(corpus) // batch_size):
            start = i * batch_size
            data_batch = corpus[start:start + batch_size]
            encoded_input = tokenizer(data_batch, padding=True, return_tensors='pt', truncation=True, max_length=100)  #
            # predict_loader = DataLoader(encoded_input, batch_size=10, num_workers=8)
            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)
            # Perform pooling. In this case, max pooling.
            batch_embeddings = cls_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings.extend(batch_embeddings)
            a = 1

    elif pooling == "max":
        tokenizer = AutoTokenizer.from_pretrained(input_model)
        model = AutoModel.from_pretrained(input_model)
        # Tokenize sentences
        encoded_input = tokenizer(corpus, padding=True, truncation=True, return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
        # Perform pooling. In this case, max pooling.
        sentence_embeddings = max_pooling(model_output, encoded_input['attention_mask'])

    else:
        # mean pools automatically
        model = SentenceTransformer(input_model)
        if type_cluster == "fast":
            sentence_embeddings = model.encode(corpus, batch_size=64, show_progress_bar=True, convert_to_tensor=True,
                                               device=device)
        else:
            sentence_embeddings = model.encode(corpus, batch_size=64, show_progress_bar=True)
        a = 1

    ###################  Store Embeddings with pickle ###################
    # Store sentences & embeddings on disc
    with open('../data/embeddings/embeddings.pkl', "wb") as fOut:
        pickle.dump({'sentences': corpus, 'embeddings': sentence_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    # Load sentences & embeddings from disc
    """
    with open('../data/embeddings/embeddings.pkl', "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_sentences = stored_data['sentences']
        stored_embeddings = stored_data['embeddings']
    """

    a = 1

    ######################### Similarity in Vectors ################

    print("Start clustering")
    start_time = time.time()

    if type_cluster == "fast":
        # fast
        # Two parameters to tune:
        # min_cluster_size: Only consider cluster that have at least 25 elements
        # threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
        clusters = util.community_detection(sentence_embeddings, min_community_size=2, threshold=0.75, batch_size=64)

        print("Clustering done after {:.2f} sec".format(time.time() - start_time))

        # Print for all clusters the top 3 and bottom 3 elements
        for i, cluster in enumerate(clusters):
            print("\nCluster {}, #{} Elements ".format(i + 1, len(cluster)))
            for sentence_id in cluster[0:3]:
                print("\t", corpus[sentence_id])
            print("\t", "...")
            for sentence_id in cluster[-3:]:
                print("\t", corpus[sentence_id])
            a = 1

    elif type_cluster == "topic":
        # topics
        print("Clustering done after {:.2f} sec".format(time.time() - start_time))
        a = 1
        
    else:
        # K Means
        num_clusters = 10
        clustering_model = KMeans(n_clusters=num_clusters, random_state=0)
        clustering_model.fit(sentence_embeddings)
        cluster_assignment = clustering_model.labels_
        print("Clustering done after {:.2f} sec".format(time.time() - start_time))

        clustered_sentences = [[] for i in range(num_clusters)]
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_sentences[cluster_id].append(corpus[sentence_id])

        for i, cluster in enumerate(clustered_sentences):
            print("Cluster ", i + 1)
            print(*cluster, sep="\n\n")
            print("")

            a = 1


if __name__ == "__main__":
    typer.run(main)

a = 1
