python ./src/router_create.py \
    --embedding_model "jinaai/jina-embeddings-v2-base-code" \
    --cache_dir "./.cache/" \
    --data_dir "./data/" \
    --output_dir "./weights/" \
    --temperature 1 \
    --similarity_metric "euclidean" \
    --visualization

