import argparse
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def load_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.
    return img

def predict_image_class(model, img, top_k):
    preds = model.predict(img)[0]
    top_indices = preds.argsort()[-top_k:][::-1]
    top_probs = preds[top_indices]
    return top_indices, top_probs

def load_class_names(json_file):
    with open(json_file, 'r') as f:
        class_names = json.load(f)
    return class_names

def main():
    parser = argparse.ArgumentParser(description='Predict image class')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('model_path', type=str, help='Path to the saved Keras model')
    parser.add_argument('--top_k', type=int, default=1, help='Number of top classes to display')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping class values to category names')
    args = parser.parse_args()
    
    model = load_model(args.model_path)
    img = load_image(args.image_path, target_size=(224, 224))
    
    top_indices, top_probs = predict_image_class(model, img, args.top_k)
    
    if args.category_names:
        class_names = load_class_names(args.category_names)
        top_class_names = [class_names[str(i)] for i in top_indices]
    else:
        top_class_names = top_indices
    
    for i in range(args.top_k):
        print(f"Class: {top_class_names[i]}, Probability: {top_probs[i]}")
        
if __name__ == '__main__':
    main()
