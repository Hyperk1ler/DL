# mnist_knn_skeleton_local.py
# 目标：给深度学习的同学的“本地数据+可运行+逐步补全”框架
# - 本地加载 MNIST（支持 IDX+gz 或 mnist.npz）
# - 跑通 sklearn KNN baseline
# - 按 TODO 逐步补全：从零实现 KNN、PCA、Top-k 精度、混淆矩阵、计时对比

import os, gzip, struct
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from time import perf_counter

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, top_k_accuracy_score

# ========= 本地数据目录（改成你的路径） =========
DATA_DIR = r"E:\CSProgram\DL\DL-CLASS-main\pj1-KNN\KNN"

# ========= 可调参数 =========
RANDOM_STATE = 42
TRAIN_N = 60000   # 设小一点（如 12000）可更快
TEST_N  = 10000   # 设小一点（如 2000）可更快
USE_PCA = True
USE_SKLEARN_BASELINE = False
N_COMP  = 50
K_LIST  = (1,3,5,7,9)
TOPK    = 5  # None 则跳过 top-k
METRIC = False
P = True
Weight = False

# ========= 工具 =========
def summarize(title, results):
    print("\n" + title)

    headers = ["k", "acc", "pred_time(s)"]
    if any("fit_time_s" in r for r in results):
        headers.insert(2, "fit_time_s")
    if TOPK is not None and any(f'top{TOPK}_acc' in r for r in results):
        headers.append(f"top{TOPK}_acc")

    print("\t".join(headers))

    for r in results:
        k_val = r.get('k', 'N/A')
        acc_val = r.get('acc', 'N/A')
        pred_time_val = r.get('pred_time_s', 'N/A')
        fit_time_val = r.get('fit_time_s', 'N/A') if 'fit_time(s)' in headers else None
        topk_val = r.get(f'top{TOPK}_acc', 'N/A') if TOPK and f'top{TOPK}_acc' in headers else None

        acc_str = f"{acc_val:.4f}" if isinstance(acc_val, float) else str(acc_val)
        pred_time_str = f"{pred_time_val:.2f}" if isinstance(pred_time_val, float) else str(pred_time_val)

        row = [str(k_val), acc_str, pred_time_str]
        if fit_time_val is not None:
            fit_time_str = f"{fit_time_val:.3f}" if isinstance(fit_time_val, float) else str(fit_time_val)
            row.insert(2, fit_time_str)
        if topk_val is not None:
            topk_str = f"{topk_val:.4f}" if isinstance(topk_val, float) else str(topk_val)
            row.append(topk_str)

        print("\t".join(row))

def _open_auto(path):
    return gzip.open(path, "rb") if path.lower().endswith(".gz") else open(path, "rb")

def _read_idx_images(path):
    with _open_auto(path) as f:
        magic = struct.unpack(">I", f.read(4))[0]
        assert magic == 2051, f"Images magic wrong: {magic}"
        n, rows, cols = struct.unpack(">III", f.read(12))
        data = np.frombuffer(f.read(n*rows*cols), dtype=np.uint8)
        return data.reshape(n, rows, cols)

def _read_idx_labels(path):
    with _open_auto(path) as f:
        magic = struct.unpack(">I", f.read(4))[0]
        assert magic == 2049, f"Labels magic wrong: {magic}"
        n = struct.unpack(">I", f.read(4))[0]
        data = np.frombuffer(f.read(n), dtype=np.uint8)
        return data

def _find_first_exist_recursive(root, names):
    # 在 root 下递归寻找 names 列表中的任意一个文件名，找到就返回完整路径
    for dirpath, _, filenames in os.walk(root):
        fl = set(filenames)
        for name in names:
            if name in fl:
                return os.path.join(dirpath, name)
    return None

def _load_idx_folder(root):
    # 支持常见命名（带/不带 .gz，dash 或 dot）
    train_img_names = ["train-images-idx3-ubyte.gz", "train-images-idx3-ubyte",
                       "train-images.idx3-ubyte.gz", "train-images.idx3-ubyte"]
    train_lab_names = ["train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte",
                       "train-labels.idx1-ubyte.gz", "train-labels.idx1-ubyte"]
    test_img_names  = ["t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte",
                       "t10k-images.idx3-ubyte.gz", "t10k-images.idx3-ubyte"]
    test_lab_names  = ["t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte",
                       "t10k-labels.idx1-ubyte.gz", "t10k-labels.idx1-ubyte"]

    p_tr_img = _find_first_exist_recursive(root, train_img_names)
    p_tr_lab = _find_first_exist_recursive(root, train_lab_names)
    p_te_img = _find_first_exist_recursive(root, test_img_names)
    p_te_lab = _find_first_exist_recursive(root, test_lab_names)
    if not all([p_tr_img, p_tr_lab, p_te_img, p_te_lab]):
        return None

    Xtr = _read_idx_images(p_tr_img).astype(np.float32)/255.0
    ytr = _read_idx_labels(p_tr_lab).astype(np.int64)
    Xte = _read_idx_images(p_te_img).astype(np.float32)/255.0
    yte = _read_idx_labels(p_te_lab).astype(np.int64)
    Xtr = Xtr.reshape(len(Xtr), -1)
    Xte = Xte.reshape(len(Xte), -1)
    return Xtr, ytr, Xte, yte

def _load_npz(root):
    # 优先找 mnist.npz；找不到就找任意 .npz 里包含所需 key 的
    candidate = _find_first_exist_recursive(root, ["mnist.npz"])
    if candidate is None:
        # 搜索任何 .npz
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(".npz"):
                    path = os.path.join(dirpath, fn)
                    try:
                        with np.load(path) as d:
                            if all(k in d for k in ["x_train","y_train","x_test","y_test"]):
                                candidate = path
                                break
                    except Exception:
                        pass
            if candidate: break
    if candidate is None:
        return None

    with np.load(candidate) as d:
        Xtr, ytr = d["x_train"], d["y_train"]
        Xte, yte = d["x_test"],  d["y_test"]
    if Xtr.ndim == 4:  # 处理通道维
        Xtr = Xtr[..., 0]
        Xte = Xte[..., 0]
    Xtr = (Xtr.astype(np.float32)/255.0).reshape(len(Xtr), -1)
    Xte = (Xte.astype(np.float32)/255.0).reshape(len(Xte), -1)
    ytr = ytr.astype(np.int64)
    yte = yte.astype(np.int64)
    return Xtr, ytr, Xte, yte

def load_mnist_local(root, train_n=TRAIN_N, test_n=TEST_N):
    # 优先用 npz，找不到再尝试 IDX
    data = _load_npz(root)
    src = "npz"
    if data is None:
        data = _load_idx_folder(root)
        src = "idx"
    if data is None:
        raise FileNotFoundError(
            f"未在 {root} 找到 mnist.npz 或 4 个 IDX 文件（train/test 的 images/labels，支持.gz）。"
        )
    Xtr, ytr, Xte, yte = data
    print(f"[load] source={src} | Train raw={Xtr.shape} Test raw={Xte.shape}")

    # 子采样
    rs = np.random.RandomState(RANDOM_STATE)
    if train_n is not None and train_n < len(Xtr):
        idx = rs.choice(len(Xtr), size=train_n, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    if test_n is not None and test_n < len(Xte):
        idx = rs.choice(len(Xte), size=test_n, replace=False)
        Xte, yte = Xte[idx], yte[idx]
    print(f"[load] after subsample -> Train={Xtr.shape} Test={Xte.shape}")
    return Xtr, ytr, Xte, yte

# ========= （可选）PCA =========
def pca_reduce(Xtr, Xte, n_components=50):
    #中心化
    mean = np.mean(Xtr, axis = 0)
    Xtr_centered = Xtr - mean
    Xte_centered = Xte - mean
    #SVD分解
    U, S, Vt = np.linalg.svd(Xtr_centered, full_matrices=False)
    Vt_reduced = Vt[:n_components]
    Xtr_pca = Xtr_centered @ Vt_reduced.T
    Xte_pca = Xte_centered @ Vt_reduced.T
    #解释方差计算
    explained_variance = (S ** 2) / (len(Xtr) - 1)
    total_var = explained_variance.sum()
    ratio = explained_variance[:n_components].sum() / total_var

    print(f"[PCA] 保留{n_components}维，解释方差：{ratio:.1%}")
    # 已实现 # 占位：先原样返回，保证可运行；完成后请替换为你的 PCA 实现
    return Xtr_pca, Xte_pca, ratio

# ========= sklearn 的 KNN baseline =========
def run_sklearn_knn(Xtr, ytr, Xte, yte, k_list=(1,3,5,7,9)):

    rows = []                               # 存储每个K值的实验结果
    for k in k_list:                        # 遍历k_list中每个K值
        base_param = {
            'n_neighbors': k,
            'algorithm': 'auto'
        }
        if METRIC:
            base_param['metric'] = 'minkowski'
        if P:
            base_param['p'] = 2
        if Weight:
            base_param['weights'] = 'uniform'
        clf = KNeighborsClassifier(**base_param)

        t0 = perf_counter()                 # 记录训练开始时间
        clf.fit(Xtr, ytr)                   # 训练
        fit_t = perf_counter() - t0         # 计算训练耗时
        t0 = perf_counter()                 # 记录预测开始时间
        y_pred = clf.predict(Xte)           # 使用训练好的模型预测测试集
        pred_t = perf_counter() - t0        # 计算预测耗时
        acc = accuracy_score(yte, y_pred)   # 计算预测准确率
        topk_acc = None                     # 初始化top-k准确率
        if TOPK is not None:
            try:
                proba = clf.predict_proba(Xte)
                topk_acc = top_k_accuracy_score(yte, proba, k=TOPK, labels=clf.classes_)
            except Exception:
                topk_acc = None
        print("fit_t:", fit_t)
        print("pred_t:", pred_t)
        print("topk_acc:", topk_acc)
        rows.append({
            'k': k,
            'acc': acc,
            #'fit_time_s': fit_t,
            'pred_time_s': pred_t,
            f'top{TOPK}_acc': topk_acc
        })
        print(rows)
    summarize("== sklearn KNN summary ==", rows)
    return

# ========= 从零实现的 KNN（学生核心练习）=========
class MyKNN:
    """最小可用 KNN（L2 距离；多数票）"""

    def __init__(self, k=5):
        self.n_classes_ = None
        self.k = k
        self.Xtr = None
        self.ytr = None
        self.classed = None

    def fit(self, X, y):
        self.Xtr = X
        self.ytr = y
        self.classed = np.unique(y)
        self.n_classes_ = len(self.classed)

    def _pairwise_distances(self, X):
        X_sq = np.sum(X**2, axis=1, keepdims=True)
        Xtr_sq = np.sum(self.Xtr**2, axis=1)
        dot_product = X @ self.Xtr.T
        dist2 = X_sq + Xtr_sq - 2 * dot_product

        return np.maximum(dist2, 0)

    def _majority_vote(self, labels_1d):
        """
        (已完成) TODO（学生完成）：多数票；平票时取“票数大且类别索引小”的
        """
        if len(labels_1d) == 0:
            return 0
        unique, counts = np.unique(labels_1d, return_counts=True)
        max_count = np.argmax(counts)
        candidate = unique[max_count]
        return np.min(candidate)

    def predict(self, X, return_proba = False, top_k = None):

        dist2 = self._pairwise_distances(X)
        idx = np.argpartition(dist2, self.k - 1, axis=1)[:, : self.k]
        neigh_labels = self.ytr[idx]
        y_pred = np.apply_along_axis(self._majority_vote, axis=1, arr=neigh_labels)

        if return_proba:
            return y_pred, self._get_proba(neigh_labels)
        elif top_k is not None:
            return y_pred, self._top_k_predictions(neigh_labels, top_k)

        return y_pred

    def _get_proba(self, neigh_labels):
        n_samples = neigh_labels.shape[0]
        proba = np.zeros((n_samples, len(self.classed)), dtype=np.float32)

        for i in range(n_samples):
            unique, counts = np.unique(neigh_labels[i], return_counts=True)
            for cls, cnt in zip(unique, counts):
                cls_idx = np.where(self.classed == cls)[0][0]
                proba[i, cls_idx] = cnt / self.k
        return proba

    def _top_k_predictions(self, neigh_labels, top_k):
        n_samples = neigh_labels.shape[0]
        top_k_preds = np.zeros((n_samples, top_k), dtype = np.int64)

        for i in range(n_samples):
            unique, counts = np.unique(neigh_labels[i], return_counts=True)

            sorted_indices = np.argsort(-counts)
            sorted_classed = unique[sorted_indices]

            if len(sorted_classed) < top_k:
                padding = np.full(top_k - len(sorted_classed), -1)
                top_k_preds[i] = np.concatenate((sorted_classed, padding))
            else:
                top_k_preds[i] = sorted_classed[:top_k]

        return top_k_preds

def run_my_knn(Xtr, ytr, Xte, yte, k_list=(1,3,5,7,9)):

    rows = []
    for k in k_list:
        model = MyKNN(k=k)
        model.fit(Xtr, ytr)
        t0 = perf_counter()
        if TOPK is not None:
            y_pred, proba = model.predict(Xte, return_proba=True)
            topk_acc = top_k_accuracy_score(yte, proba, k = TOPK, labels=model.classed)
        else:
            y_pred = model.predict(Xte)
            topk_acc = None
        pred_t = perf_counter() - t0
        acc = accuracy_score(yte, y_pred)
        row_data = {
            'k': k,
            'acc': acc,
            'pred_time_s': pred_t
        }
        if TOPK is not None:
            row_data[f'top{TOPK}_acc'] = topk_acc
        rows.append(row_data)
    summarize("== MyKNN summary ==", rows)
    return

# ========= 混淆矩阵 =========
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """
    TODO（学生完成）：
      - 用 confusion_matrix / ConfusionMatrixDisplay 画图
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print("Saved:", save_path)
    else:
        plt.show()

# ========= 主流程 =========
def main():
    Xtr, ytr, Xte, yte = load_mnist_local(DATA_DIR)

    if USE_PCA:
        Xtr, Xte, _ = pca_reduce(Xtr, Xte, n_components=50)

    if USE_SKLEARN_BASELINE:
        print("\n=== Running sklearn KNN baseline ===")
        run_sklearn_knn(Xtr, ytr, Xte, yte, k_list=(1,3,5,7,9))

        # 混淆矩阵（k=5示例）
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(Xtr, ytr)
        y_pred = clf.predict(Xte)
        plot_confusion_matrix(yte, y_pred, title="MNIST KNN (sklearn, k=5)", save_path="cm_sklearn_k5.png")
    else:
        print("\n=== Running your MyKNN (from scratch) ===")
        run_my_knn(Xtr, ytr, Xte, yte, k_list=(1,3,5,7,9))

        model = MyKNN(k=5)
        model.fit(Xtr, ytr)
        y_pred = model.predict(Xte)
        plot_confusion_matrix(yte, y_pred, title="MNIST KNN (MyKNN, k=5)", save_path="cm_mykNN_k5.png")

    print("\nDone.")

if __name__ == "__main__":
    main()
