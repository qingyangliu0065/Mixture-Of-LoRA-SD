from typing import List, Optional, Dict, Union, Tuple
import torch
import logging
import torch.nn.functional as F
import os
import json
from tqdm import tqdm

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

    
    def compute_embeddings(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.embedding_model.encode(texts)

        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)

        return embeddings


    def compute_task_centroids(self, task_examples: Dict[str, List[str]]):
        task_centroids = {}

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
            task_centroid = task_embeddings.mean(dim=0)
            task_centroids[task_name] = task_centroid
            print(task_centroids)

        # Convert to tensor
        embedding_dim = next(iter(task_centroids.values())).shape[0]
        self.task_centroids = torch.zeros((self.num_tasks, embedding_dim))

        for i, task_name in enumerate(self.tasks):
            self.task_centroids[i] = task_centroids[task_name]

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