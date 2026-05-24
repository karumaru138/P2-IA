""" MY LABELLING """
__authors__ = 'Pau Barredo | Víctor Niccolai | Sergi López'
__group__ = '20'

from utils_data import read_dataset, read_extended_dataset, crop_images
from Kmeans import KMeans


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='C:/Users/Sergi/Downloads/Codi Proj2 2025/Practica2/images', gt_json='C:/Users/Sergi/Downloads/Codi Proj2 2025/Practica2/images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
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