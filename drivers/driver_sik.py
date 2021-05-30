import numpy as np
from sklearn import preprocessing

from diffusion import Diffusion

query_path = "data/query/oxford5k_resnet_glob.npy"
gallery_path = "data/gallery/oxford5k_resnet_glob.npy"

# %%
queries = np.load(query_path)
gallery = np.load(gallery_path)
print(queries.shape)
print(gallery.shape)

path = "/home/ch/workspace/DIMA-CG-GraphicsInsight-ML/val_extraction.npz"
val_ext = np.load(path)
features, labels = val_ext["features"], val_ext["labels"]

# %%
truncation_size = 1000
kq = 10
kd = 50

n_query = len(queries)
diffusion = Diffusion(np.vstack([queries, gallery]))
offline = diffusion.get_offline_results(truncation_size, kd)
features = preprocessing.normalize(offline, norm="l2", axis=1)
scores = features[:n_query] @ features[n_query:].T
ranks = np.argsort(-scores.todense())
ranks
