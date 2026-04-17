import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import numpy as np

df = pd.read_csv("base_final.csv", encoding="utf-8-sig")
# Distribución del target
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

df["Rendimiento"].hist(bins=30, ax=axes[0])
axes[0].set_title("Distribución original")
 
np.log1p(df["Rendimiento"]).hist(bins=30, ax=axes[1])
axes[1].set_title("Log-transformado")
 
stats.probplot(df["Rendimiento"], plot=axes[2])
axes[2].set_title("QQ-Plot")
 
plt.tight_layout()
plt.show()
 
print(f"Skewness: {df['Rendimiento'].skew():.3f}")
print(f"Kurtosis: {df['Rendimiento'].kurt():.3f}")