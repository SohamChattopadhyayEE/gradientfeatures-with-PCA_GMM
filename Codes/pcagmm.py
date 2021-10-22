import numpy as np
from model_fv.fisherVectorModel import getFisherVectorModel

filename = 'E:/Job_Internships/Educazone/Dataset_Educazone_Test/features.npy'
output_path = 'E:/Job_Internships/Educazone/Dataset_Educazone_Test'
nc = 32
kd = 32

feature_space = np.load(filename)

means = []
covs = []
fvs = []
for sample in feature_space : 
    mean, cov, fv = getFisherVectorModel(sample, nc, kd)
    means.append(mean)
    covs.append(cov)
    fvs.append(fv)

path_mean = output_path + '/' + 'mean.npy'
path_cov = output_path + '/' + 'cov.npy'
path_fv = output_path + '/' + 'fv.npy'

np.save(path_mean, means)
np.save(path_cov, covs)
np.save(path_fv, fvs)


