from sklearn.manifold import TSNE

def do_tSNE(X):
    tsne = TSNE(n_components=2)
    tsne.fit_transform(X)
    print(tsne.embedding_)
    return tsne
