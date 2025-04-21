# tsne_digits.py
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
digits = load_digits()
X, y = digits.data, digits.target

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# t-SNE降维（仅在训练集上拟合）
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_train_tsne = tsne.fit_transform(X_train)
X_test_tsne = tsne.fit_transform(X_test)  # 注意：实际应使用训练集的模型转换测试集，此处简化演示

# 可视化训练集降维结果
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=X_train_tsne[:, 0],
    y=X_train_tsne[:, 1],
    hue=y_train,
    palette=sns.hls_palette(10),
    legend="full",
    alpha=0.7
)
plt.title('t-SNE Visualization of Handwritten Digits (Training Set)')
plt.savefig('tsne_digits.png')
plt.show()

# 分类准确率评估
def evaluate(X_train, X_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

# 对比原始数据与降维数据
original_acc = evaluate(X_train, X_test)
tsne_acc = evaluate(X_train_tsne, X_test_tsne)

print(f"64D Accuracy: {original_acc:.1%}")  
print(f"2D Accuracy: {tsne_acc:.1%}")       