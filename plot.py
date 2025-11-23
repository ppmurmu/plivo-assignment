import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from math import pi

# --- 1. SETUP DATA ---
# Using your actual evaluation results
data = {
    'Entity': ['CITY', 'CREDIT_CARD', 'DATE', 'EMAIL', 'PERSON_NAME', 'PHONE'],
    'Precision': [0.950, 1.000, 1.000, 1.000, 0.880, 0.932],
    'Recall':    [1.000, 1.000, 1.000, 0.759, 0.880, 0.953],
    'F1 Score':  [0.974, 1.000, 1.000, 0.863, 0.880, 0.943]
}

df = pd.DataFrame(data)

# Set global style
sns.set_theme(style="whitegrid")

# --- 2. GENERATE GROUPED BAR CHART (Standard) ---
plt.figure(figsize=(10, 6))
# Convert data to long format for Seaborn
df_melted = df.melt('Entity', var_name='Metric', value_name='Score')

# Create Bar Plot
ax = sns.barplot(
    x='Entity', y='Score', hue='Metric', 
    data=df_melted, palette="viridis"
)

plt.title('NER Performance by Entity Type', fontsize=16, fontweight='bold')
plt.ylim(0.7, 1.05)  # Zoom in to show differences
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.tight_layout()
plt.savefig('ner_performance_bar_chart.png', dpi=300)
print("✅ Saved 'ner_performance_bar_chart.png'")
plt.close()


# --- 3. GENERATE HEATMAP (Pattern Recognition) ---
plt.figure(figsize=(8, 6))
# Set Index to Entity so it appears on Y-axis
heatmap_data = df.set_index('Entity')

sns.heatmap(
    heatmap_data, 
    annot=True, 
    cmap='RdYlGn',  # Red to Green colormap
    fmt='.3f',      # 3 decimal places
    vmin=0.7, vmax=1.0, 
    linewidths=.5,
    cbar_kws={'label': 'Score'}
)

plt.title('Model Performance Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('ner_performance_heatmap.png', dpi=300)
print("✅ Saved 'ner_performance_heatmap.png'")
plt.close()


# --- 4. GENERATE RADAR CHART (The "Cool" Factor) ---
def make_radar_chart():
    categories = list(df['Entity'])
    N = len(categories)

    # Calculate angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # Close the loop

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, size=10)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.8, 0.9, 1.0], ["0.8", "0.9", "1.0"], color="grey", size=7)
    plt.ylim(0.7, 1.05)

    # Plot Precision
    values = df['Precision'].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label="Precision", color="blue")
    ax.fill(angles, values, 'blue', alpha=0.1)

    # Plot Recall
    values = df['Recall'].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label="Recall", color="red")
    ax.fill(angles, values, 'red', alpha=0.1)

    # Add Legend and Title
    plt.title('Metric Comparison Radar', size=16, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    plt.tight_layout()
    plt.savefig('ner_performance_radar.png', dpi=300)
    print("✅ Saved 'ner_performance_radar.png'")
    plt.close()

make_radar_chart()