import matplotlib.pyplot as plt
import numpy as np
# datos de la mejora 
metriques = ['Euclidiana\n(Base)', 'Manhattan\n(Millora)', 'Cosinus\n(Prova)'] # nombre de la mejora
precisio = [89.78, 91.77, 89.54] # precision de la mejroa implementada
temps = [3.00, 3.60, 5.08] # tiempo final de la ejecucion por mejora 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Impacte de la Mètrica de Distància al K-NN (K=5)', fontsize=16, fontweight='bold')

colors_acc = ['#004a87', '#84bd00', '#64748b'] # colores de la presentacion : '#004a87', '#84bd00', '#64748b'
ax1.bar(metriques, precisio, color=colors_acc, edgecolor='black')
ax1.set_title('Precisió de Classificació (%)', fontsize=14)
ax1.set_ylim(85, 95)
ax1.set_ylabel('Accuracy (%)', fontsize=12)

for i, v in enumerate(precisio):
    ax1.text(i, v + 0.2, f"{v}%", ha='center', fontweight='bold', fontsize=11)

colors_time = ['#004a87', '#84bd00', '#64748b'] # colores de la presentacion : '#004a87', '#84bd00', '#64748b'
ax2.bar(metriques, temps, color=colors_time, edgecolor='black')
ax2.set_title("Temps d'Execució (segons)", fontsize=14)
ax2.set_ylim(0, 6)
ax2.set_ylabel('Segons', fontsize=12)

for i, v in enumerate(temps):
    ax2.text(i, v + 0.1, f"{v} s", ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('grafica_resultats_knn.png', dpi=300, bbox_inches='tight')
plt.show()