"""" Load Datasets for classification problems
     Authors: Daniel Bacaicoa-Barber
              Jesús Cid-Sueiro (Original code)
              Miquel Perelló-Nieto (Original code)
"""

#importing libraries
import numpy as np

import openml
from ucimlrepo import fetch_ucirepo

import sklearn

import sklearn.datasets
import sklearn.mixture
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms

import pandas as pd
import pickle

import os

from PIL import Image




class Data_handling(Dataset):
    def __init__(self, dataset, train_size, test_size = None, valid_size = None, batch_size = 64, shuffling = False, splitting_seed = None):
        self.dataset = dataset
        self.dataset_source = None

        self.tr_size = train_size
        self.val_size = valid_size
        self.test_size = test_size

        self.weak_labels = None
        self.virtual_labels = None

        self.batch_size = batch_size

        self.shuffle = shuffling

        self.splitting_seed = splitting_seed

        openml_ids = {
            'iris': 61,  # 150 x 5 - 3 (n_samples x n_feat - n_classes)
            'pendigits': 32,  # 10992 x 17 - 10
            'glass': 41,  # 214 x 10 - 7 (really 6)
            'segment': 36,  # 2310 x 19 - 7
            'vehicle': 54,  # 846 x 19 - 4
            'vowel': 307,  # 990 x 13 - 11 <- This has cat feat to deal with
            'wine': 187,  # 178 x 14 - 3
            'abalone': 1557,  # 4177 x 9 - 3 <- Firts cat feat
            'balance-scale': 11,  # 625 x 5 - 3
            'car': 21,  # 1728 x 7 - 4 <- All categoric
            'ecoli': 39,  # 336 x 8 - 8
            'satimage': 182,  # 6435 x 37 - 6
            'collins': 478,  # 500 x 5 - 6 <- The last feat is cat (but is number as str)
            'cardiotocography': 1466,  # 2126 x 35 - 10
            'JapaneseVowels': 375,  # 9961 x 14 - 9
            'autoUniv-au6-1000': 1555,  # 1000 x 300 - 4 <- This has cat feat to deal with
            'autoUniv-au6-750': 1549,  # 750 x 300 - 4  <- This has cat feat to deal with
            'analcatdata_dmft': 469,  # 797 x 4 - 6  <- This has cat feat to deal with
            'autoUniv-au7-1100': 1552,  # 1100 x 12 - 5 <- This has cat feat to deal with
            'GesturePhaseSegmentationProcessed': 4538,  # 9873 x 32 - 5
            'autoUniv-au7-500': 1554,  # 500 x 300 - 4  <- This has cat feat to deal with
            'mfeat-zernike': 22,  # 2000 x 48 - 10
            'zoo': 62,  # 101 x 16 - 7 <- This has dichotomus feat to deal with
            'page-blocks': 30,  # 5473 x 10 - 5
            'yeast': 181,  # 1484 x 8 - 10
            'flags': 285,  # 194 x 29 - 8  <- This has cat feat to deal with
            'visualizing_livestock': 685,  # 280 x 8 - 3 <- This has cat feat to deal with
            'diggle_table_a2': 694,  # 310 x 8 - 10
            'prnn_fglass': 952,  # 214 x 9 - 6
            'confidence': 468,  # 72 x 3 - 6
            'fl2000': 477,  # 67 x 15 - 5
            'blood-transfusion': 1464,  # 748 x 4 - 2
            'banknote-authentication': 1462,  # 1372 x 4 - 2
            'cifar10': 40927, # 60000 x (32x32x3=3072) - 10
            'breast-tissue': 15,  # 699 x 9 - 2
            'cholesterol': 141,  # 10^6 x 18 - 4  <- This has cat feat to deal with
            'liver-disorders': 145,  # 10^6 x 12 - 11  <- This has cat feat to deal with
            'pasture': 294,  # 6435 x 36 - 6
            'eucalyptus': 180,  # 110393 x 54 - 7 <- This has cat feat to deal with
            'dermatology': 35,  # 366 x 34 - 6 #X = X.astype(np.float64).values should be used
            'optdigits': 28,  # 5620 x 64 - 10
            'cmc': 23,  # 1473 x 9 - 3
            }
        uci_ids = {#'breast-cancer-wisconsin': 699,  # 699 x 9 - 2
            'diabetes': 768,  # 768 x 8 - 2 #'heart-disease': 303,  # 303 x 13 - 5
            'ionosphere': 351,  # 351 x 34 - 2
            'sonar': 208,  # 208 x 60 - 2
            'parkinsons': 195,  # 195 x 22 - 2
            'seeds': 210,  # 210 x 7 - 3
            'seismic-bumps': 2584,  # 2584 x 18 - 2
            'spam': 4601,  # 4601 x 57 - 2
            'letter-recognition': 20000,  # 20000 x 16 - 26
            'bupa': 345,  # 345 x 6 - 2 (liver disorders)
            'lung-cancer': 32,  # 32 x 56 - 3
            'primary-tumor': 339,  # 339 x 17 - 21
            'mushroom': 8124,  # 8124 x 22 - 2
            'breast-cancer': 14,
            'german': 144,
            'heart': 45,
            'image':50,
            }
        
        le = sklearn.preprocessing.LabelEncoder()
        if self.dataset in self.dataset in ['mnist','kmnist','fmnist']:
            self.dataset = self.dataset.upper()
            # The standard train/test partition of this datasets will be considered 
            # but we will add the random partition
            if self.dataset == 'MNIST':
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))])
            elif self.dataset == 'KMNIST':
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1904,), (0.3475,))
                    ])
            elif self.dataset == 'FMNIST':
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860,), (0.3530,))
                    ])
            self.train_dataset = datasets.__dict__[self.dataset](
                root='Datasets/raw_datasets', 
                train=True, 
                transform=self.transform, 
                download=True)
            self.test_dataset = datasets.__dict__[self.dataset](
                root='Datasets/raw_datasets', 
                train=False, 
                transform=self.transform, 
                download=True)
            # full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
            # for the full with a random partition transforms mut be changed
            
            self.num_classes = len(np.unique(self.train_dataset.targets))
            

            self.train_num_samples = self.train_dataset.data.shape[0]
            self.test_num_samples = self.test_dataset.data.shape[0]
            
            self.train_dataset.data = self.train_dataset.data.to(torch.float32).view((self.train_num_samples,-1))
            self.test_dataset.data = self.test_dataset.data.to(torch.float32).view((self.test_num_samples,-1))
            
            self.num_features = self.train_dataset.data.shape[1]
            
            self.train_dataset.targets = self.train_dataset.targets.to(torch.long)
            self.test_dataset.targets = self.test_dataset.targets.to(torch.long)
        elif self.dataset in ['Cifar10']:
            self.dataset = self.dataset.upper()
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            self.train_dataset = datasets.__dict__[self.dataset](
                root='Datasets/raw_datasets', 
                train=True, 
                transform=self.transform, 
                download=True)
            self.test_dataset = datasets.__dict__[self.dataset](
                root='Datasets/raw_datasets', 
                train=False, 
                transform=self.transform, 
                download=True)
            self.num_classes = len(np.unique(self.train_dataset.targets))
            
            

            self.train_num_samples = self.train_dataset.data.shape[0]
            self.test_num_samples = self.test_dataset.data.shape[0]
            
            self.train_dataset.data = torch.tensor(self.train_dataset.data, dtype=torch.float32)
            self.train_dataset.data = self.train_dataset.data.permute(0, 3, 1, 2) 
            self.test_dataset.data = torch.tensor(self.test_dataset.data, dtype=torch.float32)
            self.test_dataset.data = self.test_dataset.data.permute(0, 3, 1, 2) 
            self.num_features = None
            
            
            self.train_dataset.targets = torch.tensor(self.train_dataset.targets, dtype=torch.long)
            self.test_dataset.targets = torch.tensor(self.test_dataset.targets, dtype=torch.long)

        elif self.dataset in ['Cifar100']:
            self.dataset = self.dataset.upper()
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            self.train_dataset = datasets.__dict__[self.dataset](
                root='Datasets/raw_datasets', 
                train=True, 
                transform=self.transform, 
                download=True)
            self.test_dataset = datasets.__dict__[self.dataset](
                root='Datasets/raw_datasets', 
                train=False, 
                transform=self.transform, 
                download=True)
            
            fine_to_coarse_mapping = [
                0, 0, 0, 0, 0,  # aquatic mammals
                1, 1, 1, 1, 1,  # fish
                2, 2, 2, 2, 2,  # flowers
                3, 3, 3, 3, 3,  # food containers
                4, 4, 4, 4, 4,  # fruit and vegetables
                5, 5, 5, 5, 5,  # household electrical devices
                6, 6, 6, 6, 6,  # household furniture
                7, 7, 7, 7, 7,  # insects
                8, 8, 8, 8, 8,  # large carnivores
                9, 9, 9, 9, 9,  # large man-made outdoor things
                10, 10, 10, 10, 10,  # large natural outdoor scenes
                11, 11, 11, 11, 11,  # large omnivores and herbivores
                12, 12, 12, 12, 12,  # medium-sized mammals
                13, 13, 13, 13, 13,  # non-insect invertebrates
                14, 14, 14, 14, 14,  # people
                15, 15, 15, 15, 15,  # reptiles
                16, 16, 16, 16, 16,  # small mammals
                17, 17, 17, 17, 17,  # trees
                18, 18, 18, 18, 18,  # vehicles 1
                19, 19, 19, 19, 19]   # vehicles 2]
            self.train_num_samples = self.train_dataset.data.shape[0]
            self.test_num_samples = self.test_dataset.data.shape[0]
            
            self.train_dataset.data = torch.tensor(self.train_dataset.data, dtype=torch.float32)
            self.train_dataset.data = self.train_dataset.data.permute(0, 3, 1, 2) 
            self.test_dataset.data = torch.tensor(self.test_dataset.data, dtype=torch.float32)
            self.test_dataset.data = self.test_dataset.data.permute(0, 3, 1, 2) 
            self.num_features = None

            self.train_dataset.targets = torch.tensor([fine_to_coarse_mapping[fine_idx] for fine_idx in self.train_dataset.targets], dtype=torch.long)
            self.test_dataset.targets = torch.tensor([fine_to_coarse_mapping[fine_idx] for fine_idx in self.test_dataset.targets], dtype=torch.long)

            self.num_classes = len(np.unique(self.train_dataset.targets))
            print(self.num_classes)
        elif self.dataset == 'clothing1m':
            metadata_dir = '/export/usuarios_ml4ds/danibacaicoa/ForwardBackard_losses_old/Datasets/raw_datasets/Clothing1M/'
            # Define the root directory where images are actually stored. Often the same as metadata_dir.
            # Adjust if your images are in a subdirectory like 'Clothing1M/images/'
            image_root_dir = metadata_dir

            # --- Keep your path definitions ---
            clean_label_kv_path = os.path.join(metadata_dir, 'clean_label_kv.txt')
            noisy_label_kv_path = os.path.join(metadata_dir, 'noisy_label_kv.txt')
            clean_train_key_list_path = os.path.join(metadata_dir, 'clean_train_key_list.txt')
            noisy_train_key_list_path = os.path.join(metadata_dir, 'noisy_train_key_list.txt')
            clean_val_key_list_path = os.path.join(metadata_dir, 'clean_val_key_list.txt')
            clean_test_key_list_path = os.path.join(metadata_dir, 'clean_test_key_list.txt')
            category_names_eng_path = os.path.join(metadata_dir, 'category_names_eng.txt')

            # --- Keep your helper functions load_labels, load_key_list ---
            def load_labels(filepath):
                labels = {}
                with open(filepath, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            image_path = os.path.normpath(parts[0])
                            labels[image_path] = int(parts[1]) # Store as integer
                return labels

            def load_key_list(filepath):
                keys = []
                with open(filepath, 'r') as f:
                    for line in f:
                        image_path = os.path.normpath(line.strip())
                        if image_path:
                            keys.append(image_path)
                return keys

            # --- Load category names and set num_classes ---
            category_names = []
            if os.path.exists(category_names_eng_path):
                with open(category_names_eng_path, 'r') as f:
                    category_names = [line.strip() for line in f if line.strip()]
            self.num_classes = len(category_names)
            c = self.num_classes
            print(f"Found {c} categories.")

            # --- Load all label mappings and key lists ---
            clean_labels_map = load_labels(clean_label_kv_path)
            noisy_labels_map = load_labels(noisy_label_kv_path)

            # Using ALL noisy labels as the training set base (common approach)
            # If you need the mix from your original code, adapt the logic here
            train_keys = list(noisy_labels_map.keys())
            train_labels_int = [noisy_labels_map[k] for k in train_keys]
            print(f"Using {len(train_keys)} noisy samples for training set (from noisy_label_kv.txt).")

            # Load validation and test keys
            val_keys = load_key_list(clean_val_key_list_path)
            test_keys = load_key_list(clean_test_key_list_path)

            # Filter val/test keys to ensure they have a clean label available
            test_keys = [k for k in test_keys if k in clean_labels_map]

            # Get integer labels for val/test using the clean map
            test_labels_int = [clean_labels_map[k] for k in test_keys]

            print(f"Loaded {len(val_keys)} validation samples (clean labels).")
            print(f"Loaded {len(test_keys)} test samples (clean labels).")

            # --- Define Transformations ---
            # Define separate transforms for train (with augmentation) and eval (no augmentation)
            self.train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(), # Add data augmentation for training
                # Add other augmentations if desired (e.g., RandomCrop, ColorJitter)
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.eval_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                # No augmentation for validation/testing
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            # Keep a reference to the basic transform if needed elsewhere
            self.transform = self.eval_transform

            # !!! REMOVE THE OLD NUMPY ARRAY CREATION AND LOADING LOOPS !!!
            # X_train = np.zeros(...) - REMOVED
            # for i, (key, label, _) in enumerate(self.samples): ... - REMOVED
            # y_train = np.zeros(...) - REMOVED
            # ... etc for X_test, y_test ... - REMOVED

            # --- Instantiate the Custom Dataset objects ---
            self.train_dataset = Clothing1MDataset(root_dir=image_root_dir,
                                                image_keys=train_keys,
                                                labels=train_labels_int,
                                                transform=self.train_transform)


            self.test_dataset = Clothing1MDataset(root_dir=image_root_dir,
                                                image_keys=test_keys,
                                                labels=test_labels_int,
                                                transform=self.eval_transform) # Use eval transform


            # --- Store number of samples (now derived from dataset lengths) ---
            self.train_num_samples = len(self.train_dataset)
            self.test_num_samples = len(self.test_dataset)

            # !!! REMOVE THE CIFAR-LIKE MANIPULATIONS !!!
            # self.train_dataset.data = torch.tensor(...) - REMOVED
            # self.train_dataset.targets = torch.tensor(...) - REMOVED
            # ... etc ... - REMOVED

            # Store weak labels (noisy train labels) if needed by your specific method
            # Ensure train_labels_int matches the labels in self.train_dataset
            self.weak_labels = torch.tensor(train_labels_int, dtype=torch.long)

            self.num_features = None # Input features handled by model layers




        elif self.dataset in ['clothing1m_not_efficient']:
            metadata_dir = '/export/usuarios_ml4ds/danibacaicoa/ForwardBackard_losses_old/Datasets/raw_datasets/Clothing1M/'
            clean_label_kv_path = os.path.join(metadata_dir, 'clean_label_kv.txt')
            noisy_label_kv_path = os.path.join(metadata_dir, 'noisy_label_kv.txt')
            clean_train_key_list_path = os.path.join(metadata_dir, 'clean_train_key_list.txt')
            noisy_train_key_list_path = os.path.join(metadata_dir, 'noisy_train_key_list.txt')
            clean_val_key_list_path = os.path.join(metadata_dir, 'clean_val_key_list.txt')
            clean_test_key_list_path = os.path.join(metadata_dir, 'clean_test_key_list.txt')
            category_names_eng_path = os.path.join(metadata_dir, 'category_names_eng.txt')

            def load_labels(filepath):
                """Loads image path -> label mapping from a file."""
                labels = {}

                with open(filepath, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 2:

                            image_path = os.path.normpath(parts[0])
                            labels[image_path] = int(parts[1])
                        else:

                            pass # print(f"Warning: Skipping malformed line in {filepath}: {line.strip()}")
                return labels
            def load_key_list(filepath):
                """Loads a list of image paths from a file."""
                keys = []

                with open(filepath, 'r') as f:
                    for line in f:

                        image_path = os.path.normpath(line.strip())
                        if image_path: # Ensure line is not empty
                            keys.append(image_path)
                return keys

            category_names = None

            if os.path.exists(category_names_eng_path):
                with open(category_names_eng_path, 'r') as f:
                    category_names = [line.strip() for line in f if line.strip()]


            c = len(category_names)
            
            self.samples = []

            clean_labels = load_labels(clean_label_kv_path)
            noisy_labels = load_labels(noisy_label_kv_path)
            clean_keys = load_key_list(clean_train_key_list_path)
            noisy_keys = load_key_list(noisy_train_key_list_path)
            test_keys = load_key_list(clean_test_key_list_path)

            for key in noisy_keys:
                if key in noisy_labels:
                    self.samples.append((key, np.eye(c)[noisy_labels[key]], 1)) # 1 for noisy
            for key in clean_keys:
                if key in clean_labels and key not in noisy_labels:
                    self.samples.append((key, np.eye(c)[clean_labels[key]], 0)) # 0 for clean

            for key in test_keys:
                if key in clean_labels:
                    self.samples.append((key, clean_labels[key], 0))

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),

                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats commonly used
            ])

            X_train = np.zeros((len(self.samples), 3, 224, 224))
            for i, (key, label, _) in enumerate(self.samples):
                image_path = os.path.join(metadata_dir, key)
                image = Image.open(image_path).convert('RGB')
                image = self.transform(image)
                X_train[i] = image
            y_train = np.zeros((len(self.samples), c))
            for i, (_, label, _) in enumerate(self.samples):
                y_train[i] = label
            X_test = np.zeros((len(self.test_samples), 3, 224, 224))
            for i, (key, label, _) in enumerate(self.test_samples):
                image_path = os.path.join(metadata_dir, key)
                image = Image.open(image_path).convert('RGB')
                image = self.transform(image)
                X_test[i] = image
            y_test = np.zeros((len(self.test_samples), c))
            for i, (_, label, _) in enumerate(self.test_samples):
                y_test[i] = label


            self.train_num_samples = self.train_dataset.data.shape[0]
            self.test_num_samples = self.test_dataset.data.shape[0]
            
            self.train_dataset.data = torch.tensor(self.train_dataset.data, dtype=torch.float32)
            self.train_dataset.data = self.train_dataset.data.permute(0, 3, 1, 2) 
            self.test_dataset.data = torch.tensor(self.test_dataset.data, dtype=torch.float32)
            self.test_dataset.data = self.test_dataset.data.permute(0, 3, 1, 2) 
            self.num_features = None

            self.train_dataset.targets = torch.tensor([fine_to_coarse_mapping[fine_idx] for fine_idx in self.train_dataset.targets], dtype=torch.long)
            self.test_dataset.targets = torch.tensor([fine_to_coarse_mapping[fine_idx] for fine_idx in self.test_dataset.targets], dtype=torch.long)

            self.num_classes = len(np.unique(self.train_dataset.targets))
            print(self.num_classes)
            self.weak_labels = y_train
            
        else: 
            if self.dataset in openml_ids:
                data = openml.datasets.get_dataset(openml_ids[self.dataset])
                X, y, categorical, feature_names = data.get_data(target=data.default_target_attribute)
                if any(categorical):
                    raise ValueError("TBD. For now, we don't handle categorical variables.")
                X = X.values
                y = le.fit_transform(y) #Tis encodes labels into classes [0,1,2,...,n_classes-1]
                X, y = sklearn.utils.shuffle(X, y, random_state = self.splitting_seed)
            elif self.dataset in uci_ids:
                data = fetch_ucirepo(id = uci_ids[self.dataset])
                if np.any(data.variables.type[1:]=='Categorical'):
                    raise ValueError("TBD. For now, we don't handle categorical variables.")
                X = data.data.features 
                y = data.data.targets
                X = X.values
                y = le.fit_transform(y)
                X, y = sklearn.utils.shuffle(X, y, random_state = self.splitting_seed)
            elif self.dataset == 'gmm':
                num_samples = 4000
                n_components = 4
                n_features = 3

                # Means and covariances
                means = np.array([
                    [0, 0, 0],
                    [3, 3, 3],
                    [0, -3, -3],
                    [3, 0, 0]
                    ])
                covariances = np.array([
                    3*np.eye(3),
                    1.5*np.eye(3),
                    3*np.eye(3),
                    4*np.eye(3)
                    ])

                # Mixture weights
                self.weights = np.array([0.1, 0.3, 0.5, 0.1])

                gmm = sklearn.mixture.GaussianMixture(n_components=n_components, covariance_type='full')
                gmm.weights_ = self.weights
                gmm.means_ = means
                gmm.covariances_ = covariances
                gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covariances))
                
                X, y = gmm.sample(n_samples=num_samples)

            elif self.dataset == 'hypercube':
                X, y = sklearn.datasets.make_classification(
                    n_samples=400, n_features=40, n_informative=40,
                    n_redundant=0, n_repeated=0, n_classes=4,
                    n_clusters_per_class=2,
                    weights=None, flip_y=0.0001, class_sep=1.0, hypercube=True,
                    shift=0.0, scale=1.0, shuffle=True, random_state=None)
            
            elif self.dataset == 'blobs':
                X, y = sklearn.datasets.make_blobs(
                    n_samples=400, n_features=2, centers=20, cluster_std=2,
                    center_box=(-10.0, 10.0), shuffle=True, random_state=None)
            elif self.dataset == 'blobs2':
                X, y = sklearn.datasets.make_blobs(
                    n_samples=400, n_features=4, centers=10, cluster_std=1,
                    center_box=(-10.0, 10.0), shuffle=True, random_state=None)
            elif self.dataset in uci_ids:
                raise ValueError("TBD. We still dont support UCI datasets.") 

            self.num_classes = len(np.unique(y))
            self.num_features = X.shape[1]
            if dataset == 'clothing1m':
                self.num_classes = len(np.unique(y_train))
                self.num_features = X_train.shape[1]
            else:
                X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size = self.tr_size, random_state = self.splitting_seed)
                self.num_classes = len(np.unique(y))
                self.num_features = X.shape[1]

            self.train_num_samples = X_train.shape[0]
            self.test_num_samples = X_test.shape[0]
            
            X_train = torch.from_numpy(X_train).to(torch.float32)
            X_test = torch.from_numpy(X_test).to(torch.float32)
            y_train = torch.from_numpy(y_train).to(torch.long)
            y_test = torch.from_numpy(y_test).to(torch.long)

            self.train_dataset = TensorDataset(X_train, y_train)
            self.test_dataset = TensorDataset(X_test, y_test)

            # This is done to mantain coherence between de datset classes
            self.train_dataset.data = self.train_dataset.tensors[0]
            self.train_dataset.targets = self.train_dataset.tensors[1]
            self.test_dataset.data = self.test_dataset.tensors[0]
            self.test_dataset.targets = self.test_dataset.tensors[1]
        if self.dataset not in ['clothing1m','clothing1m_not_efficient']:    
            self.train_dataset.targets = torch.eye(self.num_classes)[self.train_dataset.targets]
            self.test_dataset.targets = torch.eye(self.num_classes)[self.test_dataset.targets]

        '''#One hot encoding of the labels
        print(self.train_dataset.targets)
        self.train_dataset.targets = torch.eye(self.num_classes)[self.train_dataset.targets]
        self.test_dataset.targets = torch.eye(self.num_classes)[self.test_dataset.targets]'''

    def __getitem__(self, index):
        if self.weak_labels is None:
            x = self.train_dataset.data[index]
            y = self.train_dataset.targets[index]
            return x, y
        else:
            x = self.train_dataset.data[index]
            w = self.weak_labels[index]
            y = self.train_dataset.targets[index]
            return x, w, y
        
    def get_dataloader(self, indices = None, weak_labels = None):
        '''
        weak_labels(str): 'weak', 'virtual' or None
        '''
        #Not sure ifindices is necessary. It works this way
        if indices is None:
            indices = torch.Tensor(list(range(len(self.train_dataset)))).to(torch.long)
        if weak_labels is None: 
        #(self.weak_labels is None) & (self.virtual_labels is None):
            tr_dataset = TensorDataset(self.train_dataset.data[indices],
                                    self.train_dataset.targets[indices])
        elif weak_labels == 'virtual':
            if self.virtual_labels is None:
                print('you must provide virtual labels via include_virtual()')
                self.train_loader = None
            else:
                tr_dataset = TensorDataset(self.train_dataset.data[indices], 
                                    self.virtual_labels[indices],
                                    self.train_dataset.targets[indices])
        elif weak_labels == 'weak':
            if self.weak_labels is None:
                print('you must provide weak labels via include_weak()')
                self.train_loader = None
            else:
                tr_dataset = TensorDataset(self.train_dataset.data[indices], 
                                    self.weak_labels[indices],
                                    self.train_dataset.targets[indices])

        self.train_loader = DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                                        num_workers=0)
        self.test_loader = DataLoader(TensorDataset(
            self.test_dataset.data, self.test_dataset.targets
        ), batch_size=self.batch_size, shuffle=self.shuffle, num_workers=0)
        return self.train_loader, self.test_loader
    
    def get_data(self):
        train_x = self.train_dataset.data
        train_y = self.train_dataset.targets
        test_x = self.test_dataset.data
        test_y = self.test_dataset.targets

        return train_x, train_y, test_x, test_y
    
    def include_weak(self, z):
        if torch.is_tensor(z):
            self.weak_labels = z
        else:
            self.weak_labels = torch.from_numpy(z)
            
    def include_virtual(self, vy):
        if torch.is_tensor(vy):
            self.virtual_labels = vy
        else:
            self.virtual_labels = torch.from_numpy(vy)
