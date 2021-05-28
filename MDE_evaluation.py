


#You can use this notebook as a code template for your MDE analysis :)
# for preprocessing used: https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html#Preprocessing

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

#Load data
#I have created an h5 file with will serve as our base dataset. It contains the normalized and gene-filtered expression matrix of single-cell RNA-seq data of ~3k PBMC cells fobtained from a healthy donor.


# Loading the data into a ScanPy AnnData object with read()
base_file = 'pbmc3k_base.h5ad'
data = sc.read(base_file)

#You can get a high-level description of the data contained in the h5 file with print(). The AnnData stands for stands for annotated data matrix. This object has several attributes that we will use to access the data.
print(data)
#The obs attribute is a Pandas dataframe that contains meta information about the data, such as the cell barcodes and cell type labels.
data.obs.head(5)

#Train-Test Split
#Now we will split our data into a training and test set. To do this, we will need the individual cell barcodes, which are the row names of the obs dataframe. Once we have the split assignments, we add them as a new column to the dtaframe.

X = data.obs.index.to_list()
X_train, X_test = train_test_split(list(enumerate(X)), test_size=0.3, random_state=42)

splits=[]
train_samples = [i[1] for i in X_train]
for cell_id in X:
    if cell_id in train_samples:
        splits.append('train')
    else:
        splits.append('test')
data.obs['split'] = splits
data.obs.head(5)

#Dimensionality Reduction Baselines
#Lastly, we will perform dimensionality reduction on our training and test expression matrices using PCA, tSNE, and UMAP. The expression matrix of the entire dataset can be accessed via the X attribute. The resulting embeddings for the training and test sets are what we will be using to perform our benchmark comparisons.

train_samples_indices = [i[0] for i in X_train]
train_samples_indices.sort()
test_samples_indices = [i[0] for i in X_test]
test_samples_indices.sort()


data.X[train_samples_indices,:].shape



from sklearn.decomposition import PCA

dr_pca = PCA(n_components=2, random_state=42)
dr_pca.fit(data.X[train_samples_indices,:])

pca_train = dr_pca.transform(data.X[train_samples_indices,:])
pca_test = dr_pca.transform(data.X[test_samples_indices,:])

from sklearn.manifold import TSNE

dr_tsne = TSNE(n_components=2, random_state=42)

##tsne_train = dr_tsne.fit_transform(data.X[train_samples_indices,:])
tsne_train = dr_tsne.fit_transform(pca_train)
##tsne_test = dr_tsne.fit_transform(data.X[test_samples_indices,:])
tsne_test = dr_tsne.fit_transform(pca_test)

from umap import UMAP

dr_umap = UMAP(n_components=2, random_state=42)
##dr_umap.fit(data.X[train_samples_indices,:])
dr_umap.fit(pca_train)

##umap_train = dr_umap.transform(data.X[train_samples_indices,:])
umap_train = dr_umap.transform(pca_train)
##umap_test = dr_umap.transform(data.X[test_samples_indices,:])
umap_test = dr_umap.transform(pca_test)


# MDE Analysis - Deviation Based

import pymde

import torch

mde = pymde.preserve_neighbors(
    np.vstack([pca_train, pca_test]),
    embedding_dim=2,
    constraint=pymde.Standardized(),
    repulsive_fraction=1,
    n_neighbors=10,
    max_distance=100,
    verbose=True,
    device='cpu'
)
embedding = mde.embed(verbose=True)
In [ ]:
mde_train = embedding[:pca_train.shape[0], :]
mde_test = embedding[pca_train.shape[0]:, :]

#np.save('./mde_train.npy', mde_train)
#np.save('./mde_test.npy', mde_test)

#Visualize Embeddings with Minimum distortion embedding(MDE), PCA, TSNE, UMAP

## PCA on dataset


df = data.obs[data.obs['split'] == 'train']
df['x'] = mde_train[:,0]
df['y'] = mde_train[:,1]

plt.figure(figsize=(4, 4))
for ct in np.unique(df.cell_type):
    plt.scatter(df[df.cell_type == ct].x , df[df.cell_type == ct].y , label = ct)

plt.xlabel('MDE Dim 1')
plt.ylabel('MDE Dim 2')
#plt.legend()
plt.show()

pymde.plot(
    mde_train,
    color_by=data.obs[data.obs['split'] == 'train']['cell_type'],
    color_map='tab10',
    figsize_inches=(12, 12),
    marker_size=15,
    axis_limits=(-5,5)
)

## PCA on dataset


df = data.obs[data.obs['split'] == 'train']
df['x'] = pca_train[:,0]
df['y'] = pca_train[:,1]

plt.figure(figsize=(15, 6))
for ct in np.unique(df.cell_type):
    plt.scatter(df[df.cell_type == ct].x , df[df.cell_type == ct].y , label = ct)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

#df = df.drop([df.index[df.y.argmin()], df.index[df.y.argmax()]], axis=0)

## TSNE on dataset


df = data.obs[data.obs['split'] == 'train']
df['x'] = tsne_train[:,0]
df['y'] = tsne_train[:,1]

plt.figure(figsize=(4, 4))
for ct in np.unique(df.cell_type):
    plt.scatter(df[df.cell_type == ct].x , df[df.cell_type == ct].y , label = ct)
plt.xlabel("TSNE Dim 1")
plt.ylabel("TSNE Dim 2")
#plt.legend()
plt.show()

## UMAP on dataset

df = data.obs[data.obs['split'] == 'train']
df['x'] = umap_train[:,0]
df['y'] = umap_train[:,1]

plt.figure(figsize=(4, 4))
for ct in np.unique(df.cell_type):
    plt.scatter(df[df.cell_type == ct].x , df[df.cell_type == ct].y , label = ct)
plt.xlabel("UMAP Dim 1")
plt.ylabel("UMAP Dim 2")
#plt.legend()
plt.show()

data.obs.cell_type

## KNN Classifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

neigh_pca = KNeighborsClassifier()
df_train = data.obs[data.obs['split'] == 'train']
df_test = data.obs[data.obs['split'] == 'test']
neigh_pca.fit(pca_train,df_train['cell_type'])
y_knn_pca = neigh_pca.predict (pca_test)
# print (y_predict)
# print (df_pca_test)
print ("PCA accuracy = ", accuracy_score (df_test['cell_type'], y_knn_pca))

neigh_tsne = KNeighborsClassifier()
neigh_tsne.fit(tsne_train,df_train['cell_type'])
y_knn_tsne = neigh_tsne.predict (tsne_test)
print ("tSNE accuracy = ", accuracy_score (df_test['cell_type'], y_knn_tsne))

neigh_umap = KNeighborsClassifier()
neigh_umap.fit(umap_train,df_train['cell_type'])
y_knn_umap = neigh_umap.predict (umap_test)
print ("UMAP accuracy = ", accuracy_score (df_test['cell_type'], y_knn_umap))

neigh_mde = KNeighborsClassifier()
neigh_mde.fit(mde_train,df_train['cell_type'])
print (mde_train.shape)
print (mde_test.shape)
y_knn_mde = neigh_mde.predict (mde_test)
print ("MDE accuracy = ", accuracy_score (df_test['cell_type'], y_knn_mde))



## GMM accuracy results for pca, tsne ,umap and MDE are poorer than k neighbours.
#Also highly variable final accuracy outputs so set seed as 40 as its stochastic .Set n_component as 8 as there are 8 clusters

#from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
n_components=8
#random_state=42
# gmm=GaussianMixture(n_components, covariance_type='full', max_iter=100)
gmm=GaussianMixture(n_components, random_state=40)
from sklearn.metrics import accuracy_score

df_train = data.obs[data.obs['split'] == 'train']
df_test = data.obs[data.obs['split'] == 'test']
gmm.fit(pca_train,df_train['cell_type'])
y_predict_pca = gmm.predict (pca_test)
# print(pca_train)
# print(df_train['cell_type'])
# print(pca_test)
# print(df_test['cell_type'])
#print(y_predict_pca)

#isinstance(y_predict_pca, np.ndarray)

# # y_predict_pca_list=(y_predict_pca).tolist()
# print(y_predict_pca_list)
replacements = {
    0:'CD4 T',
    1:'CD14 Monocytes',
    2:'B',
    3:'CD8 T',
    4:'NK',
    5:'FCGR3A Monocytes',
    6:'Dendritic',
    7:'Megakaryocytes'
}

p = [replacements.get(x, x) for x in y_predict_pca]
arraypca=np.array(p)

print ("PCA accuracy = ", accuracy_score (df_test['cell_type'], arraypca))

gmm.fit(tsne_train,df_train['cell_type'])
y_predict_tsne = gmm.predict (tsne_test)
print(tsne_train)
print(tsne_test)
print(y_predict_tsne)

replacements = {
    0:'CD4 T',
    1:'CD14 Monocytes',
    2:'B',
    3:'CD8 T',
    4:'NK',
    5:'FCGR3A Monocytes',
    6:'Dendritic',
    7:'Megakaryocytes'
}

t = [replacements.get(x, x) for x in y_predict_tsne]
arraytsne=np.array(t)
print ("tSNE accuracy = ", accuracy_score (df_test['cell_type'], arraytsne))

gmm.fit(umap_train,df_train['cell_type'])
y_predict_umap = gmm.predict (umap_test)
replacements = {
    0:'CD4 T',
    1:'CD14 Monocytes',
    2:'B',
    3:'CD8 T',
    4:'NK',
    5:'FCGR3A Monocytes',
    6:'Dendritic',
    7:'Megakaryocytes'
}

u = [replacements.get(x, x) for x in y_predict_umap]


arrayumap=np.array(u)


print ("UMAP accuracy = ", accuracy_score (df_test['cell_type'],arrayumap ))

gmm.fit(mde_train,df_train['cell_type'])
print (mde_train.shape)
print (mde_test.shape)
y_predict_mde = gmm.predict (mde_test)



replacements = {
    0:'CD4 T',
    1:'CD14 Monocytes',
    2:'B',
    3:'CD8 T',
    4:'NK',
    5:'FCGR3A Monocytes',
    6:'Dendritic',
    7:'Megakaryocytes'
}

m = [replacements.get(x, x) for x in y_predict_mde]


arraymde=np.array(m)


print ("MDE accuracy = ", accuracy_score (df_test['cell_type'], arraymde))

import seaborn as sb
from sklearn import metrics
from sklearn.metrics import confusion_matrix


def Heatmap(y_true, y_preds, labels):
    """ Plot the heatmap for prediction
    y_true - the true label for each test cell for 10 iterations, should be with length 1800
    y_pred - the predict label for each test cell for 10 iterations, should be with length 1800
    labels - the order of the labels (unique_labels)
    path - file path to save plot (must end with '.eps')
    """
    confusion_array = [confusion_matrix(y_true, y_pred, labels) / 10 for y_pred in y_preds]
    sb.set(font_scale=1)

    fig, (ax1, ax2, axcb) = plt.subplots(1, 3, figsize=(12, 6), dpi=200, gridspec_kw={'width_ratios': [1, 1, 0.05]})
    ax1.get_shared_y_axes().join(ax2)
    g1 = sb.heatmap(
        confusion_array[0],
        cmap="Blues",
        cbar=False,
        ax=ax1,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"shrink": 0.5}
    )
    g1.set_ylabel('')
    g1.set_xlabel('')
    g2 = sb.heatmap(
        confusion_array[1],
        cmap="Blues",
        ax=ax2,
        cbar_ax=axcb,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"shrink": 0.5}
    )
    g2.set_ylabel('')
    g2.set_xlabel('')
    g2.set_yticks([])

    # may be needed to rotate the ticklabels correctly:
    for ax in [g1, g2]:
        tl = ax.get_xticklabels()
        ax.set_xticklabels(tl)
        tly = ax.get_yticklabels()
        ax.set_yticklabels(tly)

    plt.subplots_adjust(hspace=0.8)
    plt.subplots_adjust(left=0.3)
    # plt.xlabel('Predicted Label')
    # plt.ylabel('Actual Label')
    plt.xticks([])
    plt.show()


def plot_ROC_curve(y_true, y_score, ax, color):
    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)

    thresholds = np.linspace(0, 1, 100)
    tpr = np.zeros(100)
    fpr = np.zeros(100)
    F1_scores = np.zeros(100)
    for idx in range(100):
        y_hat = (y_score > thresholds[idx]).astype(np.int)
        tp = np.logical_and(y_hat == 1, y_true == 1).sum()
        fp = np.logical_and(y_hat == 1, y_true == 0).sum()

        tp_rate = tp / total_pos
        fp_rate = fp / total_neg

        if (tp + fp) == 0:
            ppv = 0
        else:
            ppv = tp / (tp + fp)

        if ppv == 0:
            score = 0
        else:
            score = 2 * (ppv * tp_rate) / (ppv + tp_rate)

        tpr[idx] = tp_rate
        fpr[idx] = fp_rate
        F1_scores[idx] = score

    auc = metrics.auc(fpr, tpr)
    ax.plot(fpr, tpr, linestyle='solid', color=color, linewidth=3)
    return auc, F1_scores


fig, ax = plt.subplots(figsize=(5,5))
diag_x, diag_y = [0, 1], [0, 1]
ax.plot(diag_x, diag_y, linestyle='dashed', color='#a1d76a', linewidth=3)


plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()


m = [replacements.get(x, x) for x in y_predict_mde]


unique_classes = np.array(df.cell_type.unique(), dtype='<U20')

#Heatmap(df_test['cell_type'], arraypca, unique_classes)
Heatmap(df_test['cell_type'], [y_knn_pca, arraypca], unique_classes)


Heatmap(df_test['cell_type'], [y_knn_tsne, arraytsne], unique_classes)


Heatmap(df_test['cell_type'], [y_knn_umap, arrayumap], unique_classes)


Heatmap(df_test['cell_type'], [y_knn_mde, arraymde], unique_classes)

def binarize(y, label='CD4 T'):
    binarized = (y == label).astype(np.int)
    return binarized

y_bin = binarize(df_test['cell_type']).values.astype(np.float64)
preds = [
    neigh_tsne.predict_proba(tsne_test)[:,3],
    neigh_umap.predict_proba(umap_test)[:,3],
    neigh_mde.predict_proba(mde_test)[:,3]
]


fig, ax = plt.subplots(figsize=(5,5))
diag_x, diag_y = [0, 1], [0, 1]
ax.plot(diag_x, diag_y, linestyle='dashed', color='#a1d76a', linewidth=3)


for i, pred in enumerate(preds):
    fpr, tpr, thresholds = metrics.roc_curve(y_bin, pred, pos_label=1)

    ax.plot(tpr, fpr)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()
plt.show()


fpr, tpr, thresholds = metrics.roc_curve(y_bin, preds[0], pos_label=1)

plt.plot(tpr, fpr)
plt.show()