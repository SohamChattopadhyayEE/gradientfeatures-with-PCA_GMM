from sklearn.mixture import GaussianMixture as GMM

from model_fv.utils import pca
from model_fv.utils import fisher_vector

def getFisherVectorModel(img, nc = 32, kd = 32) : 
    
    gmm = GMM(n_components=nc, covariance_type='diag')
    gmm.fit(img)
    fv = fisher_vector(img, gmm)
    return pca(gmm.means_, kd), pca(gmm.covariances_, kd), fv