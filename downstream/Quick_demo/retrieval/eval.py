import argparse  
import os  
import pandas as pd  
import numpy as np  
from sklearn.preprocessing import LabelEncoder  
from sklearn.metrics.pairwise import cosine_similarity  
import matplotlib.pyplot as plt  
  
def config():  
    parser = argparse.ArgumentParser() 
    parser.add_argument("--output_dir", type=str, default='evl_retrieval') 
    return parser.parse_args()  
  
def load_features(file_path):  
    df = pd.read_pickle(file_path)     
    features = df['feature'].tolist()  
    X = np.array(features)  
    le = LabelEncoder()  
    y = le.fit_transform(df['birads'].values)  
    return X, y  
  
def top_k_accuracy(true_labels, predicted_labels, k):  
    correct = 0  
    for true, pred in zip(true_labels, predicted_labels):  
        if true in pred[:k]:  
            correct += 1  
    return correct / len(true_labels)  


def top_k_accuracy_with_ci(true_labels, predicted_labels, k, n_bootstrap=1000):
    def top_k_acc(true, pred):
        return int(true in pred[:k])

    n_samples = len(true_labels)
    accuracies = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        true_sample = [true_labels[i] for i in indices]
        pred_sample = [predicted_labels[i] for i in indices]
        accuracy = sum(top_k_acc(true, pred) for true, pred in zip(true_sample, pred_sample)) / n_samples
        accuracies.append(accuracy)

    accuracy_mean = np.mean(accuracies)
    ci_low = np.percentile(accuracies, 2.5)
    ci_high = np.percentile(accuracies, 97.5)

    return accuracy_mean, (ci_low, ci_high)

def main(args):  
    #     encoders = ['medsam_vitb', 'lvmmed_vitb', 'lvmmed_r50','mammoclip_b2', 'mammoclip_b2', 'mama','versamammo']
    encoders = ['versamammo']
    Ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
    results = {encoder: [] for encoder in encoders}  
  
    for encoder in encoders:  
        databath_path = os.path.join('feature4retireval_versamammo', encoder+'_database.pkl')  
        query_path = os.path.join('feature4retireval_versamammo', encoder+'_query.pkl')  

        databath_features, databath_labels = load_features(databath_path)  
        query_features, query_labels = load_features(query_path)  
          
        batch_size = 1000  
        all_predicted_labels = []  
  
        for i in range(0, len(query_features), batch_size):  
            batch_queries = query_features[i:i + batch_size]  
            batch_similarities = cosine_similarity(batch_queries, databath_features)  
            top_k_indices = np.argsort(-batch_similarities, axis=1)[:, :max(Ks)]  
            batch_predicted_labels = databath_labels[top_k_indices]  
            all_predicted_labels.extend(batch_predicted_labels)  
  
        all_predicted_labels = np.array(all_predicted_labels)  
          
        # top_k_accuracies = [top_k_accuracy(query_labels, all_predicted_labels, K) for K in Ks] 
        top_k_accuracies = [top_k_accuracy_with_ci(query_labels, all_predicted_labels, K) for K in Ks]  
     
        results[encoder] = top_k_accuracies  
        print(f"{encoder} - Top K Accuracies: {top_k_accuracies}")  
  
if __name__ == "__main__":  
    args = config()  
    main(args)
