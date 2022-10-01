
from ipaddress import ip_address
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
warnings.filterwarnings('ignore') 

import fasttext

model_fname = "model_wrapper.save"
fasttext_model_fname = "fasttext_model.bin"

MODEL_NAME = "text_class_base_fasttext"

# temp dir in this folder for storing the text files that fasttext needs as input ('tempfile' didnt work)
temp_file_dir = os.path.join(os.path.dirname(__file__), 'temp_input_files')
if not os.path.exists(temp_file_dir): os.mkdir(temp_file_dir)

train_input_path = os.path.join(temp_file_dir, "_train_.txt")
valid_input_path = os.path.join(temp_file_dir, "_valid_.txt")


class Classifier(): 
    
    def __init__(self, lr=0.1, dim = 100, wordNgrams=2, **kwargs) -> None:
        self.lr = lr
        self.dim = dim
        self.wordNgrams = wordNgrams
        self.num_classes = None
        self.class_prefix = "__label__"
        self.model = None
    
    def _write_input_to_text_file(self, X, y, data_type="train"): 
        X = X. flatten()        
        fpath = train_input_path if data_type == "train" else valid_input_path            
        with open(fpath, "w", encoding='utf-8') as outf: 
            for x_, y_ in zip(X, y): 
                outf.write(f"{self.class_prefix}{y_} {x_}" + "\n")  
    
    
    def fit(self, train_X, train_y):  
        self.num_classes = len(set(train_y))
        self._write_input_to_text_file(train_X, train_y, data_type="train")   
        self.model = fasttext.train_supervised(
            input = train_input_path, 
            lr = self.lr, 
            dim = self.dim, 
            wordNgrams=self.wordNgrams, 
            epoch=100)     
        
        # quantize the model 
        # print("Quantizing model. This may take a few minutes..")
        # self.model.quantize(input = train_input_path, qnorm=True, retrain=True, cutoff=200000)   
        # print("Quantizing finished.")
        
    
    def predict_proba(self, X): 
        preds = self.model.predict(list(X.flatten()), k=-1)        
        # preds is a tuple of two elements, an (N, K) array with class labels, and an (N,K) array with associated probabilities
        classes = np.array(preds[0])
        probabilities = np.array(preds[1])        
        
        idx_2_labels_dict = {
            i: f"{self.class_prefix}{i}"
            for i in range(self.num_classes)
        }              
        pred_probs = np.zeros(shape=(X.shape[0], self.num_classes))        
        for i in range(self.num_classes): 
            label_ = idx_2_labels_dict[i]
            idxes = np.column_stack(np.where(classes == label_))
            pred_probs[idxes[:, 0], i] = probabilities[idxes[:, 0], idxes[:, 1]]
        
        pred_probs = pred_probs / pred_probs.sum(axis=1)[:, None]
        
        return pred_probs 
    
    
    def get_labels(self): 
        if self.model is not None: 
            return self.model.get_labels()
        else: return None

    def summary(self):
        self.model.get_params()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is None: return None
        
        self._write_input_to_text_file(x_test, y_test, data_type="valid")    
        test_scores = self.model.test(valid_input_path)
        precision = test_scores[1]
        recall = test_scores[2]
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score   

    
    def save(self, model_path): 
        print("Saving model...")
        self.model.save_model(os.path.join(model_path, fasttext_model_fname))
        self.model = None
        joblib.dump(self, os.path.join(model_path, model_fname))
        print("Done saving...")
        


    @classmethod
    def load(cls, model_path):         
        dtclassifier = joblib.load(os.path.join(model_path, model_fname))
        dtclassifier.model = fasttext.load_model(os.path.join(model_path, fasttext_model_fname))
        return dtclassifier


def save_model(model, model_path):
    model.save(model_path)
    

def load_model(model_path): 
    model = Classifier.load(model_path=model_path)
    return model
