import os
import yaml
import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Torch Graph Models are running on {device}")

from gnn_cp.data.data_manager import GraphDataManager
import gnn_cp.models.graph_models as graph_models
from gnn_cp.models.model_manager import GraphModelManager

def make_dataset_instances(data_dir, splits_dir, models_cache_dir, dataset_key, model_class_name, models_config):
    dataset_manager = GraphDataManager(data_dir, splits_dir)
    data = dataset_manager.get_dataset_from_key(dataset_key)
    dataset = data.data

    instances = dataset_manager.load_splits(dataset_key)

    print("Dataset Loaded Successfully!")
    print(f"Following labeled splits:")
    for c in range(dataset_manager.get_num_classes(dataset)):
        print(f"class {c}: train={(dataset.y[instances[2]['train_idx']] == c).sum()}, val={(dataset.y[instances[2]['val_idx']] == c).sum()}")

    print("====================================")
    print("Loading Models")

    print(f"Loading Models {model_class_name}")
    model_hparams = models_config.get(model_class_name, {}).get("config", {})
    optimizer_hparams = models_config.get(model_class_name, {}).get("optimizer", {})
    model_hparams.update({"n_features": dataset.x.shape[1], "n_classes": dataset.y.max().item() + 1})

    model_class = getattr(graph_models, model_class_name)
    lr = optimizer_hparams.get("lr", 0.01)
    weight_decay = optimizer_hparams.get("weight_decay", 0.0)


    for instance_idx, instance in enumerate(instances):
        model=GraphModelManager(
            model=model_class(**model_hparams), 
            optimizer_lambda=lambda model_params: torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay),
            checkpoint_address=models_cache_dir, model_name=f"{dataset_key}-ins{instance_idx}-{model_class.__name__}")
        if not model.load_model():
            print("Model not found!")
            raise FileNotFoundError(f"Model not found for instance {instance_idx}")
        else:
            pass
        model.model = model.model.to(device)
        y_pred = model.predict(dataset, test_idx=instance["test_idx"], return_embeddings=False)
        accuracy = accuracy_score(y_true=dataset.y[instance["test_idx"]].cpu().numpy(), y_pred=y_pred.cpu().numpy())
        instance.update({"model": model, "accuracy": accuracy})
    print(f"Accuracy: {np.mean([instance['accuracy'] for instance in instances])} +- {np.std([instance['accuracy'] for instance in instances])}")

    return instances