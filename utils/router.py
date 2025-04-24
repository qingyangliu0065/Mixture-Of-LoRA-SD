from typing import List, Optional, Dict, Union, Tuple
import torch
import logging
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingSimilarityRouter:
    """
    Select Lora adapter based on embedding similarity
    """

    def __init__(
        self, 
        embedding_model,
        tasks: List[str],
        temperature: float = 1.0,  # softmax(similarity / temperature)
        top_k: Optional[int] = None,  # If k, only top-k highest ranking adapters retuen non-zero weights; If None, consider all adapters
        similarity_metric: str = "cosine",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        # Init
        self.embedding_model = embedding_model
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.temperature = temperature
        self.top_k = top_k
        self.similarity_metric = similarity_metric
        self.device = device

        self.embedding_model.to(self.device)
        self.task_centroids = None
        self.task_embeddings = {}  # Store embeddings for each task
        self.all_embeddings = None  # Store concatenated embeddings
        self.all_labels = None  # Store task labels

    
    def compute_embeddings(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.embedding_model.encode(texts)

        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)

        return embeddings


    def compute_task_centroids(self, task_examples: Dict[str, List[str]]):
        task_centroids = {}
        self.task_embeddings = {}
        all_embeddings_list = []
        all_labels_list = []

        for task_name, examples in task_examples.items():
            logger.info(f"Computing centroid for task '{task_name}' from {len(examples)} examples")
            
            batch_size = 1
            all_embeddings = []

            # Progress bar
            pbar = tqdm(total=len(examples), desc=f"Task: {task_name}")

            for i in range(0, len(examples), batch_size):
                batch_texts = examples[i:i+batch_size]
                batch_embeddings = self.compute_embeddings(batch_texts)
                all_embeddings.append(batch_embeddings)
                pbar.update(len(batch_texts))

            pbar.close()

            task_embeddings = torch.cat(all_embeddings, dim=0)
            self.task_embeddings[task_name] = task_embeddings
            task_centroid = task_embeddings.mean(dim=0)
            task_centroids[task_name] = task_centroid

            # Store for visualization
            all_embeddings_list.append(task_embeddings)
            all_labels_list.extend([self.tasks.index(task_name)] * len(examples))

        # Convert to tensor
        embedding_dim = next(iter(task_centroids.values())).shape[0]
        self.task_centroids = torch.zeros((self.num_tasks, embedding_dim))

        for i, task_name in enumerate(self.tasks):
            self.task_centroids[i] = task_centroids[task_name]

        # Store concatenated embeddings and labels
        self.all_embeddings = torch.cat(all_embeddings_list, dim=0)
        self.all_labels = torch.tensor(all_labels_list)

        return self.task_centroids


    def compute_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = embeddings.to(self.task_centroids.device)

        if self.similarity_metric == "cosine":
            norm_emb = F.normalize(embeddings, p=2, dim=1)
            norm_centroids = F.normalize(self.task_centroids, p=2, dim=1)
            similarity = torch.matmul(norm_emb, norm_centroids.t())
        elif self.similarity_metric == "dot":
            similarity = torch.matmul(embeddings, self.task_centroids.t())
        elif self.similarity_metric == "euclidean":
            similarity = -torch.cdist(embeddings, self.task_centroids, p=2)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        return similarity
    

    def compute_routing_weights(self, texts: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.compute_embeddings(texts)
        similarity = self.compute_similarity(embeddings)
        scaled_similarity = similarity / self.temperature

        # Top-k selection
        if self.top_k is not None and self.top_k < self.num_tasks:
            top_k_sim, indices = torch.topk(scaled_similarity, self.top_k, dim=-1)
            mask = torch.zeros_like(scaled_similarity).scatter_(-1, indices, 1.0)
            masked_sim = torch.where(
                mask.bool(),
                scaled_similarity,
                torch.full_like(scaled_similarity, float('-inf'))
            )
            task_weights = F.softmax(masked_sim, dim=-1)
        else:
            task_weights = F.softmax(scaled_similarity, dim=-1)

        return task_weights, similarity
        

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)

        # Save task centroids
        torch.save(self.task_centroids, os.path.join(path, "router_task_centroids.pt"))

        # Save config
        config = {
            "tasks": self.tasks,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "similarity_metric": self.similarity_metric
        }
        with open(os.path.join(path, "router_config.json"), "w") as f:
            json.dump(config, f)


    @classmethod
    def load(cls, path: str, embedding_model, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        # Load config
        with open(os.path.join(path, "router_config.json"), "r") as f:
            config = json.load(f)

        # Create instance
        router = cls(
            embedding_model,
            config["tasks"],
            config["temperature"],
            config["top_k"],
            config["similarity_metric"],
            device
        )

        # Load task centroids
        router.task_centroids = torch.load(os.path.join(path, "router_task_centroids.pt"))

        logger.info(f"Router loaded from {path}")
        return router

    def visualize_embeddings(self, task_examples: Dict[str, List[str]], save_path: Optional[str] = None):
        """
        Visualize embeddings using PCA and plot them with different colors for each task.
        Also shows task centroids.
        """
        if self.all_embeddings is None or self.all_labels is None:
            raise ValueError("Embeddings not computed. Call compute_task_centroids first.")
        
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(self.all_embeddings.cpu().numpy())
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot embeddings for each task
        task_colors = plt.cm.Set1(np.linspace(0, 1, len(self.tasks)))
        for task_idx, task_name in enumerate(self.tasks):
            task_mask = self.all_labels.cpu().numpy() == task_idx
            task_embeddings = reduced_embeddings[task_mask]
            
            # Plot individual points
            plt.scatter(
                task_embeddings[:, 0], 
                task_embeddings[:, 1],
                color=task_colors[task_idx],
                label=task_name,
                alpha=0.6
            )
            
            # Plot task centroid
            if self.task_centroids is not None:
                centroid_2d = pca.transform(self.task_centroids[task_idx].cpu().numpy().reshape(1, -1))
                plt.scatter(
                    centroid_2d[0, 0],
                    centroid_2d[0, 1],
                    color=task_colors[task_idx],
                    marker='*',
                    s=200,
                    edgecolor='black',
                    linewidth=1
                )
        
        plt.title('Task Embeddings Visualization')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()