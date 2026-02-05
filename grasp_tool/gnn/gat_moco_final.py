import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import numpy as np
import pandas as pd
import random
from sklearn.cluster import KMeans
from scipy.stats import t
import math
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --- GATEncoder and ProjectionHead (from original.py) ---
class ProjectionHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x):
        return self.mlp(x)


class GATEncoder(nn.Module):  # From original.py
    def __init__(
        self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.1
    ):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = GATConv(
            hidden_channels * heads, out_channels, heads=heads, dropout=dropout
        )
        self.expected_output_dim = (
            out_channels  # For compatibility with checked.py logic if needed elsewhere
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        if batch is not None:
            graph_representation = global_mean_pool(x, batch)
        else:
            graph_representation = x.mean(dim=0)
        return x, graph_representation


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.expected_output_dim = out_channels

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        if batch is not None:
            graph_representation = global_mean_pool(x, batch)
        else:
            graph_representation = x.mean(dim=0)
        return x, graph_representation


class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=1024, m=0.999, T=0.07):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = base_encoder
        self.encoder_k = copy.deepcopy(base_encoder)

        # Use the base encoder output dimension for the projection head.
        # GATEncoder outputs out_channels, used as ProjectionHead input.
        # If expected_output_dim=128 then in_channels=128.
        # dim is the projection output dimension.
        projector_in_channels = getattr(
            base_encoder, "expected_output_dim", 128
        )  # Default to 128 if not found

        self.projector_q = ProjectionHead(
            in_channels=projector_in_channels,
            hidden_channels=256,  # original.py uses 256 in the projector hidden layer
            out_channels=dim,
        )
        self.projector_k = copy.deepcopy(self.projector_q)

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(
            self.projector_q.parameters(), self.projector_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(K, dim))
        self.queue = F.normalize(
            self.queue, dim=1
        )  # Row-wise normalization (dim=1), matching fixed.py
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    # --- Loss functions (from fixed.py) ---
    def _compute_reconstruction_loss(self, node_embeddings, edge_index, num_nodes):
        if hasattr(self, "weighted_recon_loss") and self.weighted_recon_loss:
            return self._compute_weighted_reconstruction_loss(
                node_embeddings, edge_index, num_nodes
            )
        else:
            return self._compute_basic_reconstruction_loss(
                node_embeddings, edge_index, num_nodes
            )

    def _compute_basic_reconstruction_loss(
        self, node_embeddings, edge_index, num_nodes
    ):
        # NOTE: this variant uses raw node embeddings (no normalization).
        reconstructed_adj = torch.sigmoid(
            torch.mm(node_embeddings, node_embeddings.t())
        )
        original_dense_adj = torch.zeros(
            (num_nodes, num_nodes), device=node_embeddings.device
        )
        original_dense_adj[edge_index[0], edge_index[1]] = 1
        return F.binary_cross_entropy(reconstructed_adj, original_dense_adj)

    def _compute_weighted_reconstruction_loss(
        self, node_embeddings, edge_index, num_nodes
    ):
        """Weighted reconstruction loss to mitigate class imbalance."""

        # L2-normalize node embeddings
        node_embeddings_normalized = F.normalize(node_embeddings, p=2, dim=1)

        # Pairwise similarity (logits)
        sim_scores = torch.mm(
            node_embeddings_normalized, node_embeddings_normalized.t()
        )

        # Build dense adjacency labels
        original_dense_adj = torch.zeros(
            (num_nodes, num_nodes), device=node_embeddings.device
        )
        original_dense_adj[edge_index[0], edge_index[1]] = 1

        # Compute a capped positive weight (log-scaled)
        edge_count = edge_index.size(1)
        total_pairs = num_nodes * num_nodes
        pos_weight = torch.tensor(
            min(10.0, math.log(total_pairs / max(edge_count, 1)) + 1),
            device=node_embeddings.device,
        )

        loss = F.binary_cross_entropy_with_logits(
            sim_scores,
            original_dense_adj,
            pos_weight=pos_weight,
            reduction="mean",
        )

        return loss

    def _compute_clustering_loss(
        self,
        embeddings,
        num_clusters,
        dist_type="uniform",
        input_features=None,
        batch=None,
    ):
        if dist_type == "spectral":
            k_neighbors = getattr(self, "k_neighbors", 100)
            sigma = getattr(self, "sigma", 1.0)
            if input_features is not None and batch is not None:
                graph_level_input = global_mean_pool(input_features, batch)
                return self._compute_spectral_clustering_loss(
                    graph_level_input, embeddings, k=k_neighbors, sigma=sigma
                )
            elif input_features is not None:
                print(
                    "WARNING: batch is None; pooling input_features by mean as a fallback"
                )
                graph_level_input = input_features.mean(dim=0, keepdim=True)
                return self._compute_spectral_clustering_loss(
                    graph_level_input, embeddings, k=k_neighbors, sigma=sigma
                )
            else:
                print(
                    "WARNING: input_features not provided; using embeddings as a proxy for clustering loss"
                )
                return self._compute_spectral_clustering_loss(
                    embeddings, embeddings, k=k_neighbors, sigma=sigma
                )

        batch_size = embeddings.size(0)
        if batch_size < num_clusters:
            print(
                f"WARNING: batch_size ({batch_size}) < num_clusters ({num_clusters}); using simplified clustering loss"
            )
            return self._compute_simple_clustering_loss(
                embeddings, min(num_clusters, batch_size)
            )

        try:
            embeddings_np = embeddings.detach().cpu().numpy()
            # KMeans clustering (default n_init).
            kmeans = KMeans(n_clusters=num_clusters, random_state=0, max_iter=100).fit(
                embeddings_np
            )
            pseudo_labels = kmeans.labels_

            pseudo_label_dist = torch.zeros(num_clusters, device=embeddings.device)
            if pseudo_labels is not None:  # Guard against None.
                for label in pseudo_labels:
                    pseudo_label_dist[label] += 1

            eps = 1e-8
            pseudo_label_dist = pseudo_label_dist + eps
            pseudo_label_dist /= pseudo_label_dist.sum()

            if dist_type == "t-distribution":
                embedding_mean = torch.median(embeddings, dim=1).values
                dof = 10
                scale = torch.std(embedding_mean)
                embedding_mean_np = embedding_mean.detach().cpu().numpy()
                ideal_t_dist_pdf = t.pdf(
                    embedding_mean_np,
                    dof,
                    loc=embedding_mean_np.mean(),
                    scale=scale.item(),
                )
                ideal_t_dist_tensor = torch.tensor(
                    ideal_t_dist_pdf, dtype=torch.float, device=embeddings.device
                )

                min_val = ideal_t_dist_tensor.min().item()  # Python scalar.
                max_val = ideal_t_dist_tensor.max().item()  # Python scalar.
                if max_val == min_val:  # Handle constant values.
                    max_val = min_val + eps

                ideal_t_dist_tensor = torch.histc(
                    ideal_t_dist_tensor, bins=num_clusters, min=min_val, max=max_val
                )
                ideal_t_dist_tensor = ideal_t_dist_tensor + eps
                ideal_t_dist_tensor /= ideal_t_dist_tensor.sum()
                return F.kl_div(
                    pseudo_label_dist.log(), ideal_t_dist_tensor, reduction="batchmean"
                )
            else:  # 'uniform'
                ideal_dist = (
                    torch.ones(num_clusters, device=embeddings.device) / num_clusters
                )
                return F.kl_div(
                    pseudo_label_dist.log(), ideal_dist, reduction="batchmean"
                )

        except Exception as e:
            print(
                f"ERROR: KMeans clustering failed: {e}; using simplified clustering loss"
            )
            return self._compute_simple_clustering_loss(embeddings, num_clusters)

    def _compute_simple_clustering_loss(self, embeddings, num_clusters):
        batch_size = embeddings.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        normalized_emb = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(normalized_emb, normalized_emb.t())

        mask = torch.eye(batch_size, device=embeddings.device).bool()
        similarity_matrix.masked_fill_(mask, 0)

        clustering_loss = similarity_matrix.abs().mean()
        return clustering_loss

    def _compute_spectral_clustering_loss(
        self, input_features, output_embeddings, k=100, sigma=1.0
    ):
        batch_size = output_embeddings.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=output_embeddings.device)

        input_squared_dist = torch.cdist(input_features, input_features, p=2) ** 2
        sigma_squared = 2 * (sigma**2)
        similarity = torch.exp(-input_squared_dist / sigma_squared)

        mask = torch.eye(batch_size, device=output_embeddings.device).bool()
        similarity.masked_fill_(mask, 0)

        if k < batch_size - 1:
            _, topk_indices = torch.topk(similarity, k=min(k, batch_size - 1), dim=1)
            adjacency = torch.zeros_like(similarity)
            batch_indices_arange = (
                torch.arange(batch_size, device=output_embeddings.device)
                .unsqueeze(1)
                .expand(-1, topk_indices.size(1))
            )
            adjacency[batch_indices_arange, topk_indices] = similarity[
                batch_indices_arange, topk_indices
            ]
            adjacency = 0.5 * (adjacency + adjacency.t())
        else:
            adjacency = similarity

        output_squared_dist = (
            torch.cdist(output_embeddings, output_embeddings, p=2) ** 2
        )
        loss = torch.sum(adjacency * output_squared_dist)

        edge_weights_sum = torch.sum(adjacency)
        if edge_weights_sum > 0:
            loss = loss / edge_weights_sum

        return loss

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(
            self.projector_q.parameters(), self.projector_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):  # From fixed.py (robust version)
        batch_size = keys.shape[0]
        ptr_val = int(self.queue_ptr.item())  # Scalar value from tensor.

        if ptr_val + batch_size > self.K:
            first_part = self.K - ptr_val
            self.queue[ptr_val:, :] = keys[:first_part]
            remaining = batch_size - first_part
            if remaining > 0:
                self.queue[:remaining, :] = keys[first_part:]
            new_ptr = remaining
        else:
            self.queue[ptr_val : ptr_val + batch_size, :] = keys
            new_ptr = (ptr_val + batch_size) % self.K

        self.queue_ptr[0] = new_ptr


class MoCoMultiPositive(MoCo):
    def __init__(self, base_encoder, dim=128, K=1024, m=0.999, T=0.07):
        super().__init__(base_encoder, dim, K, m, T)

    # --- forward method (batch_k_list removed) ---
    def forward(
        self,
        im_q,
        im_k_list,
        edge_index_q,
        edge_index_k_list,
        batch,
        num_clusters=None,
        dist_type="uniform",
        a=1.0,
        b=1.0,
        c=1.0,
        use_clustering=True,
    ):
        """Forward pass with multiple positive samples.

        Args:
            im_q: Node features of the query graph batch.
            im_k_list: List of node features for positive key graph batches.
            edge_index_q: Edge index for the query batch.
            edge_index_k_list: List of edge indices for the key batches.
            batch: Batch vector mapping nodes to graphs.
            num_clusters: Number of clusters used by clustering loss.
            dist_type: Target distribution for clustering loss ('uniform', 't-distribution', 'spectral').
            a: Weight for reconstruction loss.
            b: Weight for contrastive loss.
            c: Weight for clustering loss.
            use_clustering: Enable/disable clustering loss.

        Returns:
            (total_loss, reconstruction_loss, contrastive_loss, clustering_loss,
            adjusted_reconstruction, adjusted_contrastive, adjusted_clustering)
        """
        # 1. Query features
        q_node_embeddings, q = self.encoder_q(im_q, edge_index_q, batch)
        q = self.projector_q(q)
        q = F.normalize(q, dim=1)

        # 2. Positive key features
        k_features = []
        with torch.no_grad():
            self._momentum_update_key_encoder()
            for im_k, edge_index_k in zip(im_k_list, edge_index_k_list):
                _, k = self.encoder_k(im_k, edge_index_k, batch)
                k = self.projector_k(k)
                k = F.normalize(k, dim=1)
                k_features.append(k)

        # 3. Similarities to all positive samples
        l_pos_list = []
        for k in k_features:
            pos_sim = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
            l_pos_list.append(pos_sim)

        # l_pos: [batch_size, num_positives]
        # l_neg: [batch_size, K]  # K is queue size
        l_pos = torch.cat(l_pos_list, dim=1)  # shape: (batch_size, num_positives)
        l_neg = torch.einsum("nc,kc->nk", [q, self.queue.clone().detach()])

        # 4. Contrastive loss
        pos_exp = torch.exp(l_pos / self.T)  # [batch_size, num_positives]
        neg_exp = torch.exp(l_neg / self.T)
        # Numerator: sum over positives
        numerator = pos_exp.sum(dim=1)  # [batch_size]
        # Denominator: sum over positives + sum over negatives
        denominator = numerator + neg_exp.sum(dim=1)  # [batch_size]
        # loss = -log(sum(exp(pos))/sum(exp(all)))
        contrastive_loss = -torch.log(numerator / denominator).mean()

        # 5. Reconstruction loss
        reconstruction_loss = self._compute_reconstruction_loss(
            q_node_embeddings, edge_index_q, im_q.size(0)
        )

        # 6. Clustering loss (optional)
        if use_clustering:
            # Extra params for spectral clustering
            if dist_type == "spectral":
                clustering_loss = self._compute_clustering_loss(
                    q, num_clusters, dist_type, input_features=im_q, batch=batch
                )
            else:
                clustering_loss = self._compute_clustering_loss(
                    q, num_clusters, dist_type
                )
        else:
            clustering_loss = torch.tensor(0.0, device=contrastive_loss.device)

        # 7. Loss alignment
        eps = 1e-6
        adjusted_contrastive = contrastive_loss / (
            (contrastive_loss / (reconstruction_loss + eps)).detach() + eps
        )

        if use_clustering:
            adjusted_clustering = clustering_loss / (
                (clustering_loss / (reconstruction_loss + eps)).detach() + eps
            )
        else:
            adjusted_clustering = torch.tensor(0.0, device=contrastive_loss.device)

        adjusted_reconstruction = reconstruction_loss

        # 8. Total loss
        if use_clustering:
            total_loss = (
                a * adjusted_reconstruction
                + b * adjusted_contrastive
                + c * adjusted_clustering
            )
        else:
            # Only reconstruction + contrastive
            total_loss = a * adjusted_reconstruction + b * adjusted_contrastive

        # 9. Update queue (use the last positive)
        if k_features:  # Ensure k_features is not empty.
            self._dequeue_and_enqueue(k_features[-1])

        # 10. Return losses
        return (
            total_loss,
            reconstruction_loss,
            contrastive_loss,
            clustering_loss,
            adjusted_reconstruction,
            adjusted_contrastive,
            adjusted_clustering,
        )

    def forward_supcon(
        self,
        im_q,
        im_k_list,
        edge_index_q,
        edge_index_k_list,
        batch,
        num_clusters=None,
        dist_type="uniform",
        a=1.0,
        b=1.0,
        c=1.0,
        use_clustering=True,
    ):
        q_node_embeddings, q = self.encoder_q(im_q, edge_index_q, batch)
        q = self.projector_q(q)
        q = F.normalize(q, dim=1)

        k_features = []
        with torch.no_grad():
            self._momentum_update_key_encoder()
            for im_k_single, edge_index_k_single in zip(im_k_list, edge_index_k_list):
                _, k_single = self.encoder_k(im_k_single, edge_index_k_single, batch)
                k_single = self.projector_k(k_single)
                k_single = F.normalize(k_single, dim=1)
                k_features.append(k_single)

        contrastive_loss = torch.tensor(0.0, device=q.device)
        if q.size(0) > 0 and k_features:
            k_stacked = torch.stack(k_features).permute(1, 0, 2)  # [B, P, D]
            q_expanded = q.unsqueeze(1)  # [B, 1, D]
            pos_sims = (
                torch.bmm(q_expanded, k_stacked.transpose(1, 2)).squeeze(1) / self.T
            )  # [B, P]
            neg_sims = torch.matmul(q, self.queue.clone().detach().T) / self.T  # [B, K]
            logits_all = torch.cat([pos_sims, neg_sims], dim=1)
            logsum = torch.logsumexp(logits_all, dim=1, keepdim=True)
            log_probs = pos_sims - logsum
            contrastive_loss = -log_probs.mean()

        reconstruction_loss = self._compute_reconstruction_loss(
            q_node_embeddings, edge_index_q, im_q.size(0)
        )

        if use_clustering:
            clustering_loss = self._compute_clustering_loss(
                q, num_clusters, dist_type, input_features=im_q, batch=batch
            )
        else:
            clustering_loss = torch.tensor(
                0.0, device=q.device if q.numel() > 0 else torch.device("cpu")
            )

        eps = 1e-6
        adjusted_contrastive = contrastive_loss / (
            (contrastive_loss / (reconstruction_loss + eps)).detach() + eps
        )
        adjusted_reconstruction = reconstruction_loss

        if use_clustering:
            adjusted_clustering = clustering_loss / (
                (clustering_loss / (reconstruction_loss + eps)).detach() + eps
            )
            total_loss = (
                a * adjusted_reconstruction
                + b * adjusted_contrastive
                + c * adjusted_clustering
            )
        else:
            adjusted_clustering = torch.tensor(
                0.0, device=q.device if q.numel() > 0 else torch.device("cpu")
            )
            total_loss = a * adjusted_reconstruction + b * adjusted_contrastive

        if k_features:
            k_avg = torch.mean(torch.stack(k_features), dim=0)
            k_avg = F.normalize(k_avg, dim=1)
            self._dequeue_and_enqueue(k_avg)

        return (
            total_loss,
            reconstruction_loss,
            contrastive_loss,
            clustering_loss,
            adjusted_reconstruction,
            adjusted_contrastive,
            adjusted_clustering,
        )

    def forward_avg(
        self,
        im_q,
        im_k_list,
        edge_index_q,
        edge_index_k_list,
        batch,
        num_clusters=None,
        dist_type="uniform",
        a=1.0,
        b=1.0,
        c=1.0,
        use_clustering=True,
    ):
        # forward_avg is similar to forward_supcon but updates the queue differently.
        q_node_embeddings, q = self.encoder_q(im_q, edge_index_q, batch)
        q = self.projector_q(q)
        q = F.normalize(q, dim=1)

        k_features = []
        with torch.no_grad():
            self._momentum_update_key_encoder()
            for im_k_single, edge_index_k_single in zip(im_k_list, edge_index_k_list):
                _, k_single = self.encoder_k(im_k_single, edge_index_k_single, batch)
                k_single = self.projector_k(k_single)
                k_single = F.normalize(k_single, dim=1)
                k_features.append(k_single)

        contrastive_loss = torch.tensor(0.0, device=q.device)
        k_avg_for_loss = None
        if q.size(0) > 0 and k_features:
            k_avg_for_loss = torch.mean(torch.stack(k_features), dim=0)  # [B,D]
            k_avg_for_loss = F.normalize(k_avg_for_loss, dim=1)

            pos_sims_avg = (
                torch.einsum("nc,nc->n", q, k_avg_for_loss).unsqueeze(-1) / self.T
            )  # [B,1]
            neg_sims = torch.matmul(q, self.queue.clone().detach().T) / self.T  # [B, K]

            logits_all = torch.cat([pos_sims_avg, neg_sims], dim=1)  # [B, 1+K]
            logsum = torch.logsumexp(logits_all, dim=1, keepdim=True)  # [B,1]
            log_probs = pos_sims_avg - logsum  # [B,1]
            contrastive_loss = -log_probs.mean()

        reconstruction_loss = self._compute_reconstruction_loss(
            q_node_embeddings, edge_index_q, im_q.size(0)
        )

        if use_clustering:
            clustering_loss = self._compute_clustering_loss(
                q, num_clusters, dist_type, input_features=im_q, batch=batch
            )
        else:
            clustering_loss = torch.tensor(
                0.0, device=q.device if q.numel() > 0 else torch.device("cpu")
            )

        eps = 1e-6
        adjusted_contrastive = contrastive_loss / (
            (contrastive_loss / (reconstruction_loss + eps)).detach() + eps
        )
        adjusted_reconstruction = reconstruction_loss

        if use_clustering:
            adjusted_clustering = clustering_loss / (
                (clustering_loss / (reconstruction_loss + eps)).detach() + eps
            )
            total_loss = (
                a * adjusted_reconstruction
                + b * adjusted_contrastive
                + c * adjusted_clustering
            )
        else:
            adjusted_clustering = torch.tensor(
                0.0, device=q.device if q.numel() > 0 else torch.device("cpu")
            )
            total_loss = a * adjusted_reconstruction + b * adjusted_contrastive

        if k_avg_for_loss is not None:  # Use the precomputed k_avg_for_loss.
            self._dequeue_and_enqueue(k_avg_for_loss)

        return (
            total_loss,
            reconstruction_loss,
            contrastive_loss,
            clustering_loss,
            adjusted_reconstruction,
            adjusted_contrastive,
            adjusted_clustering,
        )

    # TODO: Prefer JS-distance positives when available; if insufficient, sample from same-gene graphs,
    # and only fall back to the query graph as a last resort.
    @staticmethod
    def generate_samples_gw(
        original_graphs,
        augmented_graphs,
        gene_labels,
        cell_labels,
        num_positive,
        gw_distances_df,
    ):
        """Generate positive samples using GW distances.

        Returns a list of (query_idx, [pos_idx1, pos_idx2, ...]). The first
        positive is always the augmented view of the same graph.
        """
        # (Optional) preprocessing: build gene/cell index maps
        # gene_to_indices = {}
        # cell_to_indices = {}

        # for i, (gene, cell) in enumerate(zip(gene_labels, cell_labels)):
        #     if gene not in gene_to_indices:
        #         gene_to_indices[gene] = []
        #     gene_to_indices[gene].append(i)

        #     if cell not in cell_to_indices:
        #         cell_to_indices[cell] = []
        #     cell_to_indices[cell].append(i)

        positive_samples = []
        num_graphs = len(gene_labels)
        # Build positives for each graph.
        for i in range(len(original_graphs)):
            current_positives = []

            # 1) Always include the augmented positive
            current_positives.append(i)  # augmented_graphs[i] index

            # 2) Add positives by GW distance
            target_cell = cell_labels[i]
            target_gene = gene_labels[i]

            # Filter GW distances for the current (cell, gene).
            filtered_distances = gw_distances_df[
                (gw_distances_df["target_cell"] == target_cell)
                & (gw_distances_df["target_gene"] == target_gene)
            ]

            # Take the closest (num_positive - 1) samples.
            closest_samples = filtered_distances.nsmallest(
                num_positive - 1, "gw_distance"
            )

            for _, row in closest_samples.iterrows():
                other_cell = row["cell"]
                other_gene = row["gene"]

                # Find the matching graph index.
                for j in range(len(original_graphs)):
                    if (cell_labels[j] == other_cell) and (
                        gene_labels[j] == other_gene
                    ):
                        current_positives.append(j)
                        break

            # Ensure enough positives; fall back to the query index.
            while len(current_positives) < num_positive:
                current_positives.append(i)

            positive_samples.append((i, current_positives))

        return positive_samples  # [(query_idx, [pos_idx1, pos_idx2, ...]), ...]

    @staticmethod
    def generate_samples_random_window(
        original_graphs,
        augmented_graphs,
        gene_labels,
        cell_labels,
        num_positive,
        window_size=5,
    ):
        """Generate positives by sampling same-gene graphs with similar node counts.

        The first positive is always the augmented view of the same graph.
        """
        import random

        # Precompute gene->indices and node counts.
        gene_to_indices = {}
        node_counts = {}

        # Build index mapping.
        for i, graph in enumerate(original_graphs):
            gene = gene_labels[i]

            # Number of nodes for this graph.
            node_count = (
                graph.num_real_nodes
                if hasattr(graph, "num_real_nodes")
                else sum(1 for _ in graph.x if _[2] == 0)
            )
            node_counts[i] = node_count

            # Update gene->indices mapping.
            if gene not in gene_to_indices:
                gene_to_indices[gene] = []
            gene_to_indices[gene].append(i)

        positive_samples = []

        # Build positives for each graph.
        for i in range(len(original_graphs)):
            current_positives = []

            # 1) Always include the augmented positive
            current_positives.append(i)  # augmented_graphs[i] index

            # 2) Random positives from same gene within a node-count window
            target_gene = gene_labels[i]
            target_node_count = node_counts[i]

            # All indices for the same gene.
            same_gene_indices = gene_to_indices[target_gene]

            # Candidates within the node-count window.
            candidates = []
            for idx in same_gene_indices:
                if (
                    idx != i
                    and abs(node_counts[idx] - target_node_count) <= window_size
                ):
                    candidates.append(idx)

            # If too few candidates, expand the window.
            if len(candidates) < num_positive - 1 and window_size < 20:
                extended_candidates = []
                for idx in same_gene_indices:
                    if (
                        idx != i
                        and abs(node_counts[idx] - target_node_count) <= window_size * 2
                    ):
                        extended_candidates.append(idx)
                candidates = extended_candidates

            # Randomly sample up to (num_positive - 1) candidates.
            if candidates:
                sample_size = min(len(candidates), num_positive - 1)
                sampled_indices = random.sample(candidates, sample_size)
                current_positives.extend(sampled_indices)

            # Ensure enough positives; fall back to the query index.
            while len(current_positives) < num_positive:
                current_positives.append(i)

            positive_samples.append((i, current_positives))

        return positive_samples  # [(query_idx, [pos_idx1, pos_idx2, ...]), ...]

    @staticmethod
    def generate_samples_js(
        original_graphs,
        augmented_graphs,
        gene_labels,
        cell_labels,
        num_positive,
        js_distances_df,
    ):
        """Generate positive samples using Jensen-Shannon distances."""
        print(f"Selecting positives by JS distance: num_positive={num_positive}")

        # Validate js_distances_df format
        required_columns = ["target_cell", "target_gene", "cell", "gene", "js_distance"]
        missing_columns = [
            col for col in required_columns if col not in js_distances_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"js_distances_df is missing required columns: {missing_columns}. "
                f"Expected columns: {required_columns}"
            )

        positive_samples = []

        # Build positives for each graph.
        for i in range(len(original_graphs)):
            current_positives = []

            # 1) Always include the augmented positive
            current_positives.append(i)  # augmented_graphs[i] index

            # 2) Add positives by JS distance
            target_cell = cell_labels[i]
            target_gene = gene_labels[i]

            # Filter JS distances for the current (cell, gene).
            filtered_distances = js_distances_df[
                (js_distances_df["target_cell"] == target_cell)
                & (js_distances_df["target_gene"] == target_gene)
            ]

            # Take the closest (num_positive - 1) samples.
            closest_samples = filtered_distances.nsmallest(
                num_positive - 1, "js_distance"
            )

            for _, row in closest_samples.iterrows():
                other_cell = row["cell"]
                other_gene = row["gene"]

                # Find the matching graph index.
                for j in range(len(original_graphs)):
                    if (cell_labels[j] == other_cell) and (
                        gene_labels[j] == other_gene
                    ):
                        current_positives.append(j)
                        break

            # Ensure enough positives; fall back to the query index.
            while len(current_positives) < num_positive:
                current_positives.append(i)

            positive_samples.append((i, current_positives))

        return positive_samples  # [(query_idx, [pos_idx1, pos_idx2, ...]), ...]

    @staticmethod
    def prepare_multi_positive_batch(
        original_graphs, augmented_graphs, positive_samples, batch_size
    ):
        """Yield (query_batch, positive_batches) from positive_samples."""
        # Compute number of batches and last batch size
        total_samples = len(original_graphs)
        num_batches = total_samples // batch_size
        last_batch_size = total_samples % batch_size

        # Iterate over batches
        for i in range(num_batches + (1 if last_batch_size >= 2 else 0)):
            # Decide which batch type we are processing
            if i < num_batches - 1:
                # Regular full batch
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
            elif i == num_batches - 1:
                # Last full batch (may absorb one sample)
                start_idx = i * batch_size
                if last_batch_size == 1:
                    # Absorb the last sample
                    end_idx = (i + 1) * batch_size + 1
                else:
                    # Normal end
                    end_idx = (i + 1) * batch_size
            else:
                # Final partial batch (last_batch_size >= 2)
                start_idx = num_batches * batch_size
                end_idx = total_samples

            # Batch indices
            batch_indices = list(range(start_idx, end_idx))
            # Iterate over each batch
            # for i in range(0, len(original_graphs), batch_size):
            # batch_indices = range(i, min(i + batch_size, len(original_graphs))) # (0,batch_size)

            # Query batch
            query_batch = Batch.from_data_list(
                [original_graphs[j] for j in batch_indices]
            )  # Batch handles node feature concat and edge_index offsets.
            # Positive batches
            positive_batches = []
            num_positives = len(
                positive_samples[0][1]
            )  # All samples should have the same number of positives.

            for pos_idx in range(num_positives):
                pos_batch = Batch.from_data_list(
                    [  # (j, [pos1, pos2, pos3])
                        augmented_graphs[positive_samples[j][1][0]]
                        if pos_idx == 0  # First positive uses augmented graph
                        else original_graphs[
                            positive_samples[j][1][pos_idx]
                        ]  # Similar original graph
                        for j in batch_indices
                    ]
                )
                positive_batches.append(pos_batch)
            # # Prepare positive batches (alternative implementation)
            # positive_batches = []
            # max_positives = max(len(positive_samples[j][1]) for j in batch_indices)
            # for pos_idx in range(max_positives): # iterate positives per query
            #     pos_graphs = []
            #     for j in batch_indices:
            #         query_idx, pos_indices = positive_samples[j]
            #         if pos_idx < len(pos_indices):
            #             # first positive uses augmented graph
            #             if pos_idx == 0:
            #                 pos_graphs.append(augmented_graphs[pos_indices[pos_idx]])
            #             else:
            #                 pos_graphs.append(original_graphs[pos_indices[pos_idx]]) # similar original graph
            #         else:
            #             # if not enough positives, fall back to self
            #             pos_graphs.append(original_graphs[j])

            #     pos_batch = Batch.from_data_list(pos_graphs)
            #     positive_batches.append(pos_batch)

            yield query_batch, positive_batches
