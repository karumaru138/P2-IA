""" MY LABELLING """
__authors__ = 'Pau Barredo | Víctor Niccolai | Sergi López'
__group__ = '20'

import numpy as np
import time
import matplotlib.pyplot as plt
from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval
from Kmeans import KMeans
from KNN import KNN

def benchmark_kmeans_initialization(images_data, K=4, runs=20):
    """
    Funció per avaluar el rendiment temporal i d'error (WCD) de les inicialitzacions
    """
    print(f"--- Iniciant Benchmark de K-Means (K={K}, Execucions={runs}) ---")
    
    methods = {
        'Random': {'km_init': 'random', 'tolerance': 0.001, 'max_iter': 100},
        'KMeans++': {'km_init': 'kmeans++', 'tolerance': 0.001, 'max_iter': 100}
    }
    
    for method_name, options in methods.items():
        total_wcd = 0
        total_iters = 0
        total_time = 0
        
        for i in range(runs):
            sample_data = images_data[0] 
            
            km = KMeans(sample_data, K=K, options=options)
            
            start_time = time.time()
            km.fit()
            execution_time = time.time() - start_time
            
            total_time += execution_time
            total_iters += km.num_iter
            total_wcd += km.withinClassDistance()
            
        print(f"\nResultats per {method_name}:")
        print(f"WCD Mitjana: {total_wcd / runs:.4f}")
        print(f"Iteracions Mitjanes: {total_iters / runs:.1f}")
        print(f"Temps Mitjà: {total_time / runs:.4f} segons")

def get_shape_accuracy(predicted_labels, ground_truth_labels):
    if len(predicted_labels) != len(ground_truth_labels):
        print("Error: Les llistes no tenen la mateixa longitud")
        return 0.0
   
    correctes = 0
    for pred, gt in zip(predicted_labels, ground_truth_labels):
        if pred == gt:
            correctes += 1
   
    accuracy = (correctes / len(predicted_labels)) * 100
    return accuracy

def test_knn_distances(train_imgs, train_class_labels, test_imgs, test_class_labels, k_value=5):
    """
    Testeja diferents mètriques de distància per KNN i compara la seva precisió
    """
    metrics = ['euclidean', 'cityblock', 'cosine']
    results = {}

    print(f'--- Testejant distàncies KNN (k={k_value}) ---')

    for metric in metrics:
        knn_model = KNN(train_imgs, train_class_labels)
        predicted_labels = knn_model.predict(test_imgs, k=k_value, distance_metric=metric)
        accuracy = get_shape_accuracy(predicted_labels, test_class_labels)
        results[metric] = accuracy
        print(f'Metric: {metric:10} | Precisió: {accuracy:.2f}%')
    
    return results

def get_color_accuracy(predicted_colors, ground_truth_colors):
    if len(predicted_colors) != len(ground_truth_colors):
        print("Error: Les llistes no tenen la mateixa longitud")
        return 0.0
   
    total_solapament = 0.0
    for pred, gt in zip(predicted_colors, ground_truth_colors):
        set_pred = set(pred) if isinstance(pred, list) else {pred}
        set_gt = set(gt) if isinstance(gt, list) else {gt}
       
        if len(set_pred) == 0 and len(set_gt) == 0:
            solapament = 1.0  
        elif len(set_pred.union(set_gt)) == 0:
            solapament = 0.0
        else:
            interseccio = len(set_pred.intersection(set_gt))
            unio = len(set_pred.union(set_gt))
            solapament = interseccio / unio
       
        total_solapament += solapament
   
    accuracy = (total_solapament / len(predicted_colors)) * 100
    return accuracy

def Retrieval_by_color(images, color_labels, query, color_percentages=None):
    if isinstance(query, str):
        query = [query]

    retrieved = []
    for i, img_colors in enumerate(color_labels):
        if all(q in img_colors for q in query):
            if color_percentages is not None:
                score = sum(color_percentages[i].get(q, 0) for q in query)
                retrieved.append((images[i], score))
            else:
                retrieved.append(images[i])
    
    if color_percentages is not None:
        retrieved.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in retrieved]
    
    return retrieved

def Retrieval_by_shape(images, shape_labels, query, knn_confidence=None):
    retrieved = []
    for i, shape in enumerate(shape_labels):
        if shape == query:
            if knn_confidence is not None:
                score = knn_confidence[i]
                retrieved.append((images[i], score))
            else:
                retrieved.append(images[i])
                
    if knn_confidence is not None:
        retrieved.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in retrieved]
        
    return retrieved

def Retrieval_combined(images, shape_labels, color_labels, shape_query, color_query, knn_confidence=None, color_percentages=None):
    if isinstance(color_query, str):
        color_query = [color_query]
        
    retrieved = []
    for i in range(len(images)):
        shape_match = (shape_labels[i] == shape_query)
        color_match = all(q in color_labels[i] for q in color_query)
        
        if shape_match and color_match:
            if knn_confidence is not None and color_percentages is not None:
                shape_score = knn_confidence[i]
                color_score = sum(color_percentages[i].get(q, 0) for q in color_query)
                combined_score = (shape_score + color_score) / 2
                retrieved.append((images[i], combined_score))
            else:
                retrieved.append(images[i])
                
    if knn_confidence is not None and color_percentages is not None:
        retrieved.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in retrieved]
        
    return retrieved

def avaluar_thresholds_reals(cropped_imgs, ground_truth_colors, max_K=7):
    """
    Executa K-Means per a diferents llindars (thresholds), calcula la K mitjana 
    i l'accuracy de color real per a l'informe.
    """
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
    num_imatges = len(cropped_imgs)
    
    print(f"\n--- Iniciant Avaluació de Thresholds sobre {num_imatges} imatges ---")
    
    print(f"{'Threshold':<12} | {'K Mitjana':<12} | {'Color Accuracy':<15}")
    print("-" * 45)

    for thr in thresholds:
        total_k = 0
        predicted_colors_for_thr = []
        
        for i, img in enumerate(cropped_imgs):
            km_eval = KMeans(img, K=1)
            km_eval.find_bestK(max_K=max_K, threshold=thr)
            best_k = km_eval.K
            total_k += best_k
            
            km_final = KMeans(img, K=best_k)
            km_final.fit()
            
            colors = KMeans.get_colors(km_final.centroids)
            predicted_colors_for_thr.append(colors)
            
        k_mitjana = total_k / num_imatges
        accuracy = get_color_accuracy(predicted_colors_for_thr, ground_truth_colors)
        
        print(f"{thr:<12.2f} | {k_mitjana:<12.2f} | {accuracy:.2f}%")

if __name__ == '__main__':

    ruta_images = 'C:/Users/Usuario/Documents/GitHub/P2-IA/images'
    ruta_gt = 'C:/Users/Usuario/Documents/GitHub/P2-IA/images/gt.json'
    
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels = read_dataset(root_folder=ruta_images, gt_json=ruta_gt)

    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    avaluar_thresholds_reals(cropped_images, color_labels, max_K=7)

    print('\n======================================================')
    distance_results = test_knn_distances(train_imgs, train_class_labels, test_imgs, test_class_labels, k_value=5)

    print('\n======================================================')
    benchmark_kmeans_initialization(cropped_images, K=5, runs=20)

    print('\n======================================================')
    print("--- 1. Calculant prediccions de forma (KNN) ---")
    modelo_knn = KNN(train_imgs, train_class_labels)
    predicted_shapes = modelo_knn.predict(test_imgs, k=5, distance_metric='euclidean')

    print("--- 2. Calculant prediccions de color (KMeans) ---")
    predicted_colors = []
    for img in test_imgs:
        km = KMeans(img, K=3) 
        km.fit()
        colors = KMeans.get_colors(km.centroids)
        predicted_colors.append(colors)

    print("\n--- 3. Filtrant resultats i generant gràfica ---")
    forma_buscada = "Flip Flops"
    color_buscado = "Red"

    imatges_recuperades = Retrieval_combined(
        images=test_imgs,
        shape_labels=predicted_shapes, 
        color_labels=predicted_colors, 
        shape_query=forma_buscada,
        color_query=color_buscado
    )

    if len(imatges_recuperades) > 0:
        print(f"S'han trobat {len(imatges_recuperades)} imatges. Obrint finestra de Matplotlib...")
        visualize_retrieval(
            imatges_recuperades, 
            min(5, len(imatges_recuperades)),
            title=f"Resultats cerca combinada: {color_buscado} {forma_buscada}"
        )
        plt.show() 
    else:
        print(f"No s'han trobat resultats per '{color_buscado} {forma_buscada}'.")