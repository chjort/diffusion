import numpy as np
import faiss


class Diffusion:
    def __init__(self, features, affinity="cosine"):
        super(Diffusion, self).__init__()
        features = np.array(features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(
                "`features` must have rank 2. Found rank {}.".format(features.ndim)
            )

        d = features.shape[1]
        if affinity == "cosine":
            self.nn = faiss.faiss.IndexFlatIP(d)
        elif affinity == "euclidean":
            self.nn = faiss.faiss.IndexFlatL2(d)
        self.affinity = affinity
