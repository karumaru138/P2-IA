""" MY LABELLING """
__authors__ = 'Pau Barredo | Víctor Niccolai | Sergi López'
__group__ = '20'

from utils_data import read_dataset, read_extended_dataset, crop_images
from Kmeans import KMeans
from KNN import KNN
import numpy as np
import time

def benchmark_kmeans_initialization(images_data, K=4, runs=20):

    """
    Funcio per veure els diferents benchmark de cada versio
    """

    print(f"--- Iniciant Benchmark de K-Means (K={K}, Execucions={runs}) ---")
    
    methods = {
        'Random': {'km_init': 'random', 'tolerance': 0.001, 'max_iter': 100},
        'KMeans++': {'km_init': 'kmeans++', 'tolerance': 0.001, 'max_iter': 100} # Qualsevol string que no sigui 'first' o 'random' activa el teu K-means++
    }
    
    for method_name, options in methods.items():
        total_wcd = 0
        total_iters = 0
        total_time = 0
        
        for i in range(runs):
            # Agafem una imatge petita com a mostra o aplanem un subset per anar ràpid
            sample_data = images_data[0] # Utilitza la primera imatge per la prova, o un subset
            
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
    Testeja diferents metriques de distancia per KNN i compara la seva precisió
    """

    metrics=['euclidean', 'cityblock', 'cosine']
    results = {}

    print(f'--- Testejant distancies KNN (k={k_value}) ---')

    for metric in metrics:
        knn_model = KNN(train_imgs, train_class_labels)

        predicted_labels = knn_model.predict(test_imgs, k=k_value, distance_metric=metric)

        accuracy = get_shape_accuracy(predicted_labels, test_class_labels)
        results[metric] =  accuracy

        print(f'Metric: {metric:10} | Precisio: {accuracy:.2f}%')
    
    return results

if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='C:/Users/Usuario/Documents/GitHub/P2-IA/images', gt_json='C:/Users/Usuario/Documents/GitHub/P2-IA/images/gt.json')

    print('\nExecutatnt Test de millores de les Distance Metrics...')
    distance_results = test_knn_distances(train_imgs, train_class_labels, test_imgs, test_class_labels, k_value=5)

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here

    print('\nTest de temps de execucio...')
    benchmark_kmeans_initialization(cropped_images, K=5, runs=20)


def get_color_accuracy(predicted_colors, ground_truth_colors):
 
    if len(predicted_colors) != len(ground_truth_colors):
        print("Error: Les llistes no tenen la mateixa longitud")
        return 0.0
   
    total_solapament = 0.0
   
    for pred, gt in zip(predicted_colors, ground_truth_colors):
       
        set_pred = set(pred) if isinstance(pred, list) else {pred}
        set_gt = set(gt) if isinstance(gt, list) else {gt}
       
        if len(set_pred) == 0 and len(set_gt) == 0:
            solapament = 1.0  # Ambdós buits, es considera encert
        elif len(set_pred.union(set_gt)) == 0:
            solapament = 0.0
        else:
           
            interseccio = len(set_pred.intersection(set_gt))
            unio = len(set_pred.union(set_gt))
            solapament = interseccio / unio
       
        total_solapament += solapament
   
    accuracy = (total_solapament / len(predicted_colors)) * 100
    return accuracy

def retrieval_by_color(images, color_labels, query, color_percentages=None):

    if isinstance(query, str):
        query = [query]

    retrieved = []

    for i, img_colors in enumerate(color_labels):
        #Comprova si tots els colors de la query son als labes d'aquesta imatge
        if all(q in img_colors for q in query):
            if color_percentages is not None:
                #Sumar el percentatge dels colors queried per crear la sorting score
                score = sum(color_percentages[i].get(q,0) for q in query)
                retrieved.append((images[i], score))
            else:
                retrieved.append(images[i])
    
    if color_percentages is not None:
        retrieved.sort(key = lambda x: x[1], reverse=True)
        return [item[0] for item in retrieved]
    
    return retrieved

