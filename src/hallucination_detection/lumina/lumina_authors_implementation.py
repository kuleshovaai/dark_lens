import torch
from torch.nn import functional as F


class LUMINA():
    def __init__(self, model, tokenizer, kernel='cosine', lam=0.5, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.lam = lam
        self.device = device
        
        if kernel == "cosine":
            self.kernel = self.__cosine_kernel
        elif kernel == "rbf":
            self.kernel = self.__rbf_kernel
        else:
            raise ValueError("Invalid kernel. Choose 'cosine' or 'rbf'.")
    
    @staticmethod
    @torch.no_grad()
    def __compute_entropy(probs):
        """Compute entropy with numerical stability."""
        return -torch.sum(probs * torch.log(probs.clamp(min=1e-8)), dim=-1)
    
    @torch.no_grad()
    def __compute_ipr(self, hid_prob, ans_prob, ans_ids):
        """
        Compute Information Processing Rate (IPR) efficiently using vectorized operations.
        """
        T = ans_prob.shape[0]
        num_layers = len(hid_prob)
        
        # Stack all layer probabilities: (num_layers, T, vocab_size)
        hid_prob_stacked = torch.stack(hid_prob)
        
        # Get max predictions for each token position: (T,)
        max_ids = torch.argmax(ans_prob, dim=-1)
        
        # Compute entropy for all layers and tokens at once: (num_layers, T)
        entropy = self.__compute_entropy(hid_prob_stacked)
        
        # Compute weights (inverse entropy): (num_layers, T)
        weights = 1.0 / (entropy + 1e-8)
        
        # Layer indices (1-based): (num_layers, 1)
        layer_indices = torch.arange(1, num_layers + 1, device=self.device).unsqueeze(1)
        
        # Extract probabilities for max_ids across all layers: (num_layers, T)
        batch_indices = torch.arange(T, device=self.device).unsqueeze(0).expand(num_layers, -1)
        hid_max_probs = hid_prob_stacked[
            torch.arange(num_layers, device=self.device).unsqueeze(1),
            batch_indices,
            max_ids.unsqueeze(0).expand(num_layers, -1)
        ]
        ans_max_probs = ans_prob[batch_indices[0], max_ids]  # (T,)
        
        # Compute ratios: (num_layers, T)
        ratios = 1 - torch.clamp(hid_max_probs / ans_max_probs.unsqueeze(0), max=1.0)
        
        # Weighted layer ratios: (num_layers, T)
        weighted_ratios = ratios * layer_indices
        
        # Sum over layers and normalize: (T,)
        total_weighted_ratio = weighted_ratios.sum(dim=0)
        total_weight = (layer_indices * weights).sum(dim=0)
        
        # Extract answer probabilities: (T,)
        ans_token_probs = ans_prob[batch_indices[0], ans_ids]
        
        # Final IPR computation: (T,)
        ipr = (total_weighted_ratio / total_weight) * (ans_token_probs / ans_max_probs)
        
        return ipr.cpu()
    
    @torch.no_grad()
    def __get_topk_embeddings_and_probs(self, probs, k, embedding_layer):
        """
        Efficiently extract top-k tokens and embeddings for all time steps.
        """
        # Get top-k for all tokens at once: (T, k)
        topk_probs, topk_ids = torch.topk(probs, k, dim=-1)
        
        # Get embeddings: (T, k, d)
        embeddings = embedding_layer(topk_ids)
        
        return topk_probs.float(), embeddings
    
    @staticmethod
    def __rbf_kernel(x, y, sigma=1.0):
        """RBF (Gaussian) kernel."""
        dist_sq = torch.cdist(x, y, p=2).pow(2)  # More efficient than manual broadcasting
        return torch.exp(-dist_sq / (2 * sigma ** 2)).float()
    
    @staticmethod
    def __cosine_kernel(x, y, eps=1e-8):
        """
        Compute cosine similarity kernel between two sets of vectors.
        
        Args:
            x: Tensor of shape (N, d)
            y: Tensor of shape (M, d)
            eps: Small value to avoid division by zero
            
        Returns:
            A (N, M) tensor where each entry [i, j] = (1 + cosine_similarity(x[i], y[j])) / 2
        """
        # Normalize vectors
        x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
        y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
        
        # Compute cosine similarity and shift to [0, 1]
        return (1 + torch.matmul(x_norm, y_norm.T)) / 2
    
    @torch.no_grad()
    def __compute_mmd(self, p_prob, q_prob, embedding_layer, k=100, **kernel_kwargs):
        """
        Compute Maximum Mean Discrepancy using vectorized operations.
        """
        # Get top-k for both distributions
        p_top_k, p_embed = self.__get_topk_embeddings_and_probs(p_prob, k, embedding_layer)
        q_top_k, q_embed = self.__get_topk_embeddings_and_probs(q_prob, k, embedding_layer)
        
        T = p_prob.shape[0]
        mmd_scores = []
        
        # Process in batches to avoid memory issues with very long sequences
        batch_size = 32
        for i in range(0, T, batch_size):
            end_idx = min(i + batch_size, T)
            
            # Compute kernel matrices for batch
            K_pp = torch.stack([
                p_top_k[t] @ self.kernel(p_embed[t], p_embed[t], **kernel_kwargs) @ p_top_k[t].T
                for t in range(i, end_idx)
            ])
            
            K_qq = torch.stack([
                q_top_k[t] @ self.kernel(q_embed[t], q_embed[t], **kernel_kwargs) @ q_top_k[t].T
                for t in range(i, end_idx)
            ])
            
            K_pq = torch.stack([
                p_top_k[t] @ self.kernel(p_embed[t], q_embed[t], **kernel_kwargs) @ q_top_k[t].T
                for t in range(i, end_idx)
            ])
            
            # MMD² = E[k(p,p)] + E[k(q,q)] - 2E[k(p,q)]
            mmd_batch = K_pp + K_qq - 2 * K_pq
            mmd_scores.append(mmd_batch)
        
        return torch.cat(mmd_scores).cpu()
    
    @torch.no_grad()
    def __get_ans(self, logits, input_ids, prefix_ids, hidden_states=None):
        """
        Extract the response portion and compute probabilities.
        
        Returns:
            - probs: Probability distribution over vocabulary for response tokens
            - targets: Target token IDs for the response
            - hidden_states: (Optional) Hidden states for response tokens from all layers
        """
        # Focus only on the response portion
        start = prefix_ids.shape[-1]
        
        # Compute probabilities (shift left for next-token prediction alignment)
        probs = F.softmax(logits[:, start-1:-1, :], dim=-1).squeeze(0).float()
        
        # Extract target tokens
        targets = input_ids[:, start:].squeeze(0)
        
        if hidden_states is not None:
            # Extract hidden states for response portion
            hid_states = [hidden_state[:, start-1:-1, :].squeeze(0) for hidden_state in hidden_states]
            return probs, targets, hid_states
        
        return probs, targets
    
    def __build_input(self, prompt, response):
        """
        Build input tensors for model inference.
        
        Args:
            prompt: The prompt with context
            response: The generated response
            
        Returns:
            input_ids: Tokenized full input (prompt + response)
            prefix_ids: Tokenized prompt only
        """
        # Truncate prompt to avoid context length issues
        messages = [{"role": "user", "content": prompt[:12000]}]
        
        # Apply chat template
        prefix = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Combine prefix and response
        input_text = prefix + ' ' + response
        
        # Tokenize
        input_ids = self.tokenizer([input_text], return_tensors="pt").input_ids.to(self.device)
        prefix_ids = self.tokenizer([prefix], return_tensors="pt").input_ids.to(self.device)
        
        return input_ids, prefix_ids
    
    @torch.no_grad()
    def predict(self, prompt_w_context, prompt_w_random_context, response):
        """
        Predict hallucination score for a given response.
        
        Args:
            prompt_w_context: Prompt with correct retrieved context
            prompt_w_random_context: Prompt with random/irrelevant context
            response: Generated response to evaluate
            
        Returns:
            hallucination_score: Combined score (higher = more likely hallucination)
            mmd: External context utilization score
            ipr: Internal knowledge utilization score
        """
        # Build inputs
        input_w_context_ids, prefix_w_context_ids = self.__build_input(prompt_w_context, response)
        input_w_random_ids, prefix_w_random_ids = self.__build_input(prompt_w_random_context, response)
        
        # Forward pass with correct context
        outputs_w_context = self.model(
            input_ids=input_w_context_ids,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False  # Disable KV cache to save memory
        )
        
        # Forward pass with random context
        outputs_w_random = self.model(
            input_ids=input_w_random_ids,
            return_dict=True,
            output_hidden_states=False,  # Don't need hidden states for random context
            use_cache=False
        )
        
        # Extract logits and hidden states
        logits_w_context = outputs_w_context['logits']
        hidden_states_w_context = outputs_w_context['hidden_states'][1:]  # Skip input embeddings
        logits_w_random = outputs_w_random['logits']
        
        # Get embedding layer
        embedding_layer = self.model.get_input_embeddings()
        
        # Extract response probabilities and hidden states
        prob_w_context, answer_ids, answer_hid = self.__get_ans(
            logits_w_context, input_w_context_ids, prefix_w_context_ids, hidden_states_w_context
        )
        prob_w_random, _ = self.__get_ans(
            logits_w_random, input_w_random_ids, prefix_w_random_ids
        )
        
        # Compute MMD (external context score)
        mmd = self.__compute_mmd(prob_w_context, prob_w_random, embedding_layer, k=100)
        
        # Compute logit lens for all layers
        logit_lens_res = []
        for hid in answer_hid:
            # Apply final layer norm and project to vocabulary
            if hasattr(self.model.model, 'language_model'):
                lens_logits = self.model.lm_head(self.model.model.language_model.norm(hid))
            else:
                lens_logits = self.model.lm_head(self.model.model.norm(hid))
            
            logit_lens_res.append(F.softmax(lens_logits, dim=-1))
        
        # Compute IPR (internal knowledge score)
        ipr = self.__compute_ipr(logit_lens_res, prob_w_context, answer_ids)
        
        # Combine scores
        hallucination_score = self.lam * ipr - (1 - self.lam) * mmd
        
        return hallucination_score, mmd, ipr