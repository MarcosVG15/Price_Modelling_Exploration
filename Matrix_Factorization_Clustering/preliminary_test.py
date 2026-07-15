import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD


_HERE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(_HERE, "..", "..", "data_files", "all_feature_data_Headphones.csv")

df = pd.read_csv(path, skiprows=[0], low_memory=False)
df = df.drop(columns=df.columns[0])


numeric = df.apply(pd.to_numeric, errors="coerce")
present = df.notna() & ~numeric.eq(0)          # zero-valued numerics -> absent
mock_data = present.to_numpy(dtype=np.uint8)
X_raw = csr_matrix(mock_data)

density = X_raw.nnz / (X_raw.shape[0] * X_raw.shape[1])
print(f"Original Matrix Shape: {X_raw.shape}")
print(f"Presence density: {density:.2%}")

tfidf = TfidfTransformer()
X_tfidf = tfidf.fit_transform(X_raw)


n_components = 50
svd = TruncatedSVD(n_components=n_components, random_state=42)
svd.fit(X_tfidf)

explained_variance = svd.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)


print("\n--- JUSTIFICATION METRICS FOR REPORT ---")
target_ks = [5, 10, 30, 50]
for k in target_ks:
    if k <= n_components:
        # Sum the variance of the first k components
        var_explained = cumulative_variance[k-1] * 100
        print(f"k = {k:2d} latent components explain {var_explained:.2f}% of the total variance.")

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Number of Latent Components (k)', fontsize=12)
ax1.set_ylabel('Individual Variance Explained', color=color, fontsize=12)
ax1.bar(range(1, n_components + 1), explained_variance, color=color, alpha=0.6, label='Individual')
ax1.tick_params(axis='y', labelcolor=color)

# Plot 2: Cumulative variance explained (Line Plot)
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Cumulative Variance Explained (Cumulative sum)', color=color, fontsize=12)
ax2.plot(range(1, n_components + 1), cumulative_variance, color=color, marker='o', linewidth=2, label='Cumulative')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Scree Plot & Cumulative Explained Variance', fontsize=14, fontweight='bold')
fig.tight_layout()  
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


feature_names = df.columns.tolist()

print("\n--- REVEALING THE LATENT COMPONENTS ---")
for comp_idx in range(5):  # Let's look at the first 5 components
    print(f"\nComponent {comp_idx + 1}:")
    
    component_weights = svd.components_[comp_idx]
    top_indices = np.argsort(np.abs(component_weights))[::-1][:10]
    
    # Print the top 10 features and their weights
    for idx in top_indices:
        weight = component_weights[idx]
        print(f"  {feature_names[idx]:<30} : {weight:+.4f}")