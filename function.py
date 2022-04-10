import numpy as np

def SVD(S,f):
    U, sigma, Vt = np.linalg.svd(S)
    U = U[ : , : f ]
    sigma = np.diag(sigma[0:f])
    X = np.dot(U, sigma ** 0.5)
    return X

def cos_sim(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        # print(dot_product)
        normA += a ** 2
        normB += b ** 2
    if normB == 0.0 or normA == 0.0:
        return 0
    else:
        return dot_product / ((normA ** 0.5) * (normB ** 0.5))

def lapsim(dd):
    mm = np.zeros(dd.shape)
    for i in range(mm.shape[0]):
        for j in range(mm.shape[1]):
            mm[i, j] = dd[i, j] / np.sqrt(np.sum(dd[i, :]) * np.sum(dd[j, :]))
    return mm

def tpr_fpr_precision_recall(true, pred):
    tp, fp, tn, fn = 0, 0, 0, 0
    index = list(reversed(np.argsort(pred)))
    tpr = []
    fpr = []
    precision = []
    recall = []
    for i in range(pred.shape[0]):
        if true[int(index[i])] == 1:
            tp += 1
        else:
            fp += 1
        if np.sum(true) == 0:
            tpr.append(0)
            fpr.append(0)
            precision.append(0)
            recall.append(0)
        else:
            tpr.append(tp / np.sum(true))
            fpr.append(fp / (true.shape[0] - np.sum(true)))
            precision.append(tp / (tp + fp))
            recall.append(tp / np.sum(true))
    return tpr, fpr, precision, recall

def equal_len_list(a):
    row_len = []
    for i in a:
        row_len.append(len(i))
    min_len = min(row_len)
    equal_len_a = []
    for i in a:
        tem_list = []
        multi = len(i)/min_len
        for j in range(min_len-1):
            tem_list.append(i[int(j*multi)])
        tem_list.append(i[-1])
        equal_len_a.append(tem_list)
    return equal_len_a

class IKNN:
    def __init__(self, k) -> None:
        self._k = k

    def fit(self, _X):
        self.X = _X.copy()
        return self

    def neighbors(self, i) -> np.ndarray:
        drugs_sim = self.X[i]
        values = np.sort(drugs_sim)[::-1][1:self._k + 1]
        indexes = np.argsort(drugs_sim)[::-1][1:self._k + 1]
        return (indexes, values)

def preprocess(K, drug_mat, target_mat, Y, a,rs,miu=0.7):
    (n, m) = Y.shape
    Yd = np.zeros((n, m))
    Yt = np.zeros((n, m))
    iknn = IKNN(K)
    weights = np.zeros(K)

    iknn.fit(drug_mat)
    for d in range(n):
        (indexes, values) = iknn.neighbors(d)
        z = np.sum(values)
        if z==0:
            continue
        for i in range(K):
            weights[i] = (miu ** i) * values[i]
            Yd[d] += weights[i] * Y[indexes[i]] / z

    iknn.fit(target_mat)
    for t in range(m):
        (indexes, values) = iknn.neighbors(t)
        z = np.sum(values)
        if z==0:
            continue
        for i in range(K):
            weights[i] = (miu ** i) * values[i]
            Yt[:, t] += weights[i] * Y[:, indexes[i]] / z

    for i in range(n):
        for j in range(m):
            Y[i][j] = max(Y[i][j],( rs*(Yd[i][j] + Yt[i][j])+(1-rs)*a[i][j] )/ 2)
    return Y
