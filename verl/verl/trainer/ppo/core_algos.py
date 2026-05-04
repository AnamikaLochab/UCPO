# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from collections import defaultdict

import numpy as np
import torch

import verl.utils.torch_functional as verl_F

import numpy as np
import torch

def _canon_index(index, batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Return a contiguous int64 tensor of shape [batch_size] on `device`.
    Accepts: list/tuple, np.ndarray (any dtype), or torch.Tensor.
    Maps arbitrary labels (strings, objects) to [0..G-1] via np.unique.
    """
    if torch.is_tensor(index):
        idx_t = index.to(device=device).long().view(-1)
        if idx_t.numel() != batch_size:
            raise ValueError(f"index len {idx_t.numel()} != batch {batch_size}")
        return idx_t

    # Anything non-tensor → NumPy
    arr = np.asarray(index)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.shape[0] != batch_size:
        raise ValueError(f"index len {arr.shape[0]} != batch {batch_size}")

    # If not numeric (or mixed), factorize to ints
    if arr.dtype == np.object_ or not np.issubdtype(arr.dtype, np.integer):
        # Map arbitrary labels to contiguous ints (inverse)
        _, inverse = np.unique(arr.astype(object), return_inverse=True)
        arr = inverse.astype(np.int64, copy=False)
    else:
        arr = arr.astype(np.int64, copy=False)

    return torch.from_numpy(arr).to(device)

class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns

from collections import defaultdict

import numpy as np
import torch
import random

from scipy.special import comb


def calc_adv(val, k):
    c = len(np.where(val==1)[0])
    n = len(val)
    rho = 1 - comb(n-c, k) / comb(n, k)
    sigma = np.sqrt(rho * (1 - rho))
    adv_p = (1 - rho) / (sigma + 1e-6)
    adv_n = (1 - rho - comb(n-c-1, k-1)/comb(n-1,k-1)) / (sigma + 1e-6)
    new_val = np.where(val==1, adv_p, val)
    new_val = np.where(new_val==0, adv_n, new_val)
    return new_val

def compute_advantage_pass_K(token_level_rewards, response_mask, index, K=4, old_log_prob=None):
    scores = token_level_rewards.sum(dim=-1)
    
    id2score = defaultdict(list)
    uid2sid = defaultdict(list)
    id2mean = {}
    id2std = {}

    stats = {
        "Hq_correct": [],
        "mc_correct": [],
        "num_correct": [],
        "Hq_incorrect": [],
        "num_incorrect": [],
    }

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i].detach().item())
            uid2sid[index[i]].append(i)

        # ---- Optional: precompute seq_logp + lengths for q/Hq stats ----
        if old_log_prob is not None:
            # seq logprob over response tokens
            seq_logp = (old_log_prob * response_mask).sum(dim=-1)  # [B]
            lengths = (response_mask > 0).sum(dim=-1).clamp(min=1) # [B]

        for uid in id2score.keys():
            reward = np.array(id2score[uid])
            adv = calc_adv(reward, K)
            print(uid2sid[uid])
            for i in range(len(uid2sid[uid])):
                scores[uid2sid[uid][i]] = adv[i]

            # ---- Stats: q/Hq on correct & incorrect subsets within this uid ----
            if old_log_prob is not None:
                idxs = uid2sid[uid]
                idxs_t = torch.tensor(idxs, device=token_level_rewards.device, dtype=torch.long)

                g_scores = token_level_rewards.sum(dim=-1)[idxs_t]  # [n]
                g_logp = seq_logp[idxs_t]                           # [n]
                g_len = lengths[idxs_t]                             # [n]

                corr_mask = (g_scores > 0)

                # Correct-set stats
                if corr_mask.sum() > 1:
                    log_pi = g_logp[corr_mask] / g_len[corr_mask]
                    log_q = log_pi - torch.logsumexp(log_pi, dim=0)
                    q = log_q.exp()
                    Hq = -(q * log_q).sum()
                    stats["Hq_correct"].append(Hq.detach().cpu())
                    stats["mc_correct"].append(q.max().detach().cpu())
                    stats["num_correct"].append(torch.tensor(float(corr_mask.sum().item())))

                elif corr_mask.sum() == 1:
                    stats["num_correct"].append(torch.tensor(1.0))

                # Incorrect-set stats
                inc_mask = ~corr_mask
                if inc_mask.sum() > 1:
                    log_pi = g_logp[inc_mask] / g_len[inc_mask]
                    log_q = log_pi - torch.logsumexp(log_pi, dim=0)
                    q = log_q.exp()
                    Hq = -(q * log_q).sum()
                    stats["Hq_incorrect"].append(Hq.detach().cpu())
                    stats["num_incorrect"].append(torch.tensor(float(inc_mask.sum().item())))

                elif inc_mask.sum() == 1:
                    stats["num_incorrect"].append(torch.tensor(1.0))

    scores = scores.unsqueeze(-1) * response_mask
    
    return scores, scores, stats


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,   # [B, T]
    response_mask: torch.Tensor,         # [B, T]
    old_log_prob: torch.Tensor,          # [B, T]  (needed for q/H diagnostics)
    index: np.ndarray,                   # len B group IDs (prompt IDs)
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """
    GRPO outcome advantages + diversity diagnostics computed on correct subset.

    Returns:
        advantages: [B, T]
        stats: dict[str, list[torch.Tensor]]  # 0-d CPU tensors
    """
    device = token_level_rewards.device
    B = token_level_rewards.size(0)

    # outcome reward per response
    scores = token_level_rewards.sum(dim=-1)  # [B]

    # seq logp per response (for q/H diagnostics)
    seq_logp = (old_log_prob * response_mask).sum(dim=-1)  # [B]
    lengths = (response_mask > 0).sum(dim=-1).clamp(min=1) # [B]

    stats = {
        # these are what you plot to show DivGRPO > GRPO
        "Hq_correct": [],
        "mc_correct": [],
        "num_correct": [],
        # optional incorrect diagnostics
        "Hq_incorrect": [],
        "num_incorrect": [],
    }

    # group -> indices
    id2idx = defaultdict(list)
    for i in range(B):
        id2idx[index[i]].append(i)

    advantages = torch.zeros_like(scores)  # [B]

    with torch.no_grad():
        # ---- GRPO advantage (unchanged) ----
        for gid, idxs in id2idx.items():
            idxs_t = torch.tensor(idxs, device=device, dtype=torch.long)
            g = scores[idxs_t]  # [n]

            if g.numel() == 1:
                mean_g = torch.tensor(0.0, device=device)
                std_g = torch.tensor(1.0, device=device)
            else:
                mean_g = g.mean()
                std_g = g.std(unbiased=True)

            if norm_adv_by_std_in_grpo:
                advantages[idxs_t] = (g - mean_g) / (std_g + epsilon)
            else:
                advantages[idxs_t] = (g - mean_g)

        # ---- Diversity diagnostics (correct subset) ----
        for gid, idxs in id2idx.items():
            idxs_t = torch.tensor(idxs, device=device, dtype=torch.long)

            g_scores = scores[idxs_t]
            g_logp = seq_logp[idxs_t]
            g_len = lengths[idxs_t]

            corr_mask = (g_scores > 0)
            if corr_mask.sum() > 1:
                log_pi = g_logp[corr_mask] / g_len[corr_mask]  # length-normalized
                log_q = log_pi - torch.logsumexp(log_pi, dim=0)
                q = log_q.exp()
                Hq = -(q * log_q).sum()
                mc = q.max()

                stats["Hq_correct"].append(Hq.detach().cpu())
                stats["mc_correct"].append(mc.detach().cpu())
                stats["num_correct"].append(torch.tensor(float(corr_mask.sum().item())))
            elif corr_mask.sum() == 1:
                # still log num_correct for your plots
                stats["num_correct"].append(torch.tensor(1.0))

            # optional incorrect stats if you want them
            inc_mask = ~corr_mask
            if inc_mask.sum() > 1:
                log_pi = g_logp[inc_mask] / g_len[inc_mask]
                log_q = log_pi - torch.logsumexp(log_pi, dim=0)
                q = log_q.exp()
                Hq = -(q * log_q).sum()
                stats["Hq_incorrect"].append(Hq.detach().cpu())
                stats["num_incorrect"].append(torch.tensor(float(inc_mask.sum().item())))
            elif inc_mask.sum() == 1:
                    stats["num_incorrect"].append(torch.tensor(1.0))

        advantages = advantages.unsqueeze(-1) * response_mask  # [B, T]

    return advantages, advantages, stats

def compute_fgrpo_advantage(
    token_level_rewards: torch.Tensor,   # [B, T]
    response_mask: torch.Tensor,         # [B, T]
    old_log_prob: torch.Tensor,          # [B, T]  # kept for interface compatibility
    index,                               # group IDs
    gamma: float = 0.5,
    epsilon: float = 1e-8,
):
    """
    F-GRPO advantage: scale standard GRPO advantages by a group-level focal weight.

        A_i^{F-GRPO} = w(X) * A_i^{GRPO}
        w(X) = (1 - mu_pos)^gamma
        mu_pos = X / N

    where:
      - X = number of correct samples in the group
      - N = group size

    Important:
      - The SAME weight is applied to all trajectories in the group
      - This preserves gradient direction within the group and only scales magnitude
    """
    device = token_level_rewards.device
    stats = {
        "num_correct": [],
        "num_incorrect": [],
        "mu_pos": [],
        "focal_weight": [],
        "adv_delta_mean": [],
        "adv_delta_max": [],
        "adv_delta_min": [],
        "adv_delta_ratio": [],
        "adv_original_mean": [],
        "adv_modified_mean": [],
    }

    B = token_level_rewards.size(0)
    idx_t = _canon_index(index, batch_size=B, device=device)
    scores = token_level_rewards.sum(dim=-1)  # [B]

    # 1. Standard GRPO advantages
    advantages = torch.zeros_like(scores)
    unique_groups = torch.unique(idx_t)

    for gid in unique_groups:
        gmask = (idx_t == gid)
        group_scores = scores[gmask]
        if group_scores.numel() > 1:
            mean_g = group_scores.mean()
            std_g = group_scores.std(unbiased=True)
            advantages[gmask] = (group_scores - mean_g) / (std_g + epsilon)
        else:
            advantages[gmask] = group_scores - group_scores.mean()

    # Snapshot original GRPO advantages
    original_advantages = advantages.clone()

    # 2. Apply F-GRPO group-level focal scaling
    for gid in unique_groups:
        gmask = (idx_t == gid)
        gidx = torch.nonzero(gmask, as_tuple=False).squeeze(-1)
        group_scores = scores[gidx]

        N = group_scores.numel()
        X = (group_scores > 0).sum().float()
        num_incorrect = N - int(X.item())

        mu_pos = X / max(N, 1)
        weight = (1.0 - mu_pos).clamp(min=0.0) ** gamma

        # track stats
        stats["num_correct"].append(torch.tensor(float(X.item())))
        stats["num_incorrect"].append(torch.tensor(float(num_incorrect)))
        stats["mu_pos"].append(mu_pos.detach().cpu())
        stats["focal_weight"].append(weight.detach().cpu())

        new_advantages = original_advantages[gidx] * weight
        delta = new_advantages - original_advantages[gidx]

        stats["adv_delta_mean"].append(delta.mean().detach().cpu())
        stats["adv_delta_max"].append(delta.max().detach().cpu())
        stats["adv_delta_min"].append(delta.min().detach().cpu())

        orig_A_mag = original_advantages[gidx].abs().mean()
        if orig_A_mag > 0.01:
            stats["adv_delta_ratio"].append(
                (delta.abs().mean() / orig_A_mag).detach().cpu()
            )

        advantages[gidx] = new_advantages.detach()

    # 3. Global sanity stats
    stats["adv_original_mean"].append(original_advantages.abs().mean().detach().cpu())
    stats["adv_modified_mean"].append(advantages.abs().mean().detach().cpu())

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages, stats

#change start


def compute_dgrpo_advantage_outcome_A_mult(
    token_level_rewards: torch.Tensor,   # [B, T]
    response_mask: torch.Tensor,         # [B, T]
    old_log_prob: torch.Tensor,          # [B, T]
    index,                               # group IDs
    tau: float = 0.1,                    # Diversity strength (Correct)
    alpha: float = 0.1,                  # Sharpening strength (Incorrect)
    epsilon: float = 1e-8,
):
    device = token_level_rewards.device
    stats = {
        "Hq_correct": [],
        "mc_correct": [],
        "num_correct": [],
        "Hq_incorrect": [],
        "num_incorrect": [],
    }

    B = token_level_rewards.size(0)
    idx_t = _canon_index(index, batch_size=B, device=device) 
    # 1. Basic Setup
    scores = token_level_rewards.sum(dim=-1)                  # [B]
    seq_logp = (old_log_prob * response_mask).sum(dim=-1)     # [B]
    # print("Alpha: ",alpha, " Tau: ", tau)
    # 2. Compute Standard GRPO Advantage (Baseline)
    # We do the standard Mean/Std normalization first.
    advantages = torch.zeros_like(scores)
    unique_groups = torch.unique(idx_t)

    for gid in unique_groups:
        gmask = (idx_t == gid)
        group_scores = scores[gmask]
        
        # Standard GRPO Normalization
        if group_scores.numel() > 1:
            mean_g = group_scores.mean()
            std_g = group_scores.std(unbiased=False)
            # Standard Advantage A_i
            advantages[gmask] = (group_scores - mean_g) / (std_g + epsilon)
        else:
            # Fallback for singleton groups
            advantages[gmask] = group_scores - group_scores.mean()

    # 3. Apply Distributional Corrections to A
    # Now we add the diversity/sharpening terms directly to A.
    
    for gid in unique_groups:
        gidx = torch.nonzero(idx_t == gid, as_tuple=False).squeeze(-1)
        
        # Get data for this group
        group_logp = seq_logp[gidx]
        group_scores = scores[gidx]
        
        # Dynamic Temperature (Crucial for Softmax stability)
        # Using avg_len to scale log-probs to ~[-2, 0] range
        # avg_len = response_mask[gidx].sum(dim=1).float().mean().clamp(min=1.0)
        lengths = (response_mask[gidx] > 0).sum(dim=-1)
        # --- PART A: CORRECT SUBGROUP (Diversity) ---
        corr_mask = (group_scores > 0)
        if corr_mask.sum() > 1:
            # 1. Get Log-Probs of correct items
            valid_idx = gidx[corr_mask]
            log_pi = group_logp[corr_mask]
            lens=lengths[corr_mask]
            # 2. Compute q (Distribution)
            # Scale by temp to prevent Softmax collapse
            log_pi_scaled = log_pi/lens#avg_len
            log_q = log_pi_scaled - torch.logsumexp(log_pi_scaled, dim=0)
            log_q = log_q.detach()
            q = log_q.exp()
            
            Hq = -(q * log_q).sum()
            # ---- STATS (Correct Set) ----
            stats["Hq_correct"].append(Hq.detach().cpu())
            stats["mc_correct"].append(q.max().detach().cpu())
            stats["num_correct"].append(
                torch.tensor(corr_mask.sum().item(), dtype=torch.float32)
            )
        

            # 3. Compute Surprisal
            surprisal = -log_q
            # surprisal = -log_pi_scaled
            # Term = -log q - H(q)
            # If -log q > H(q) (Rare), term is Positive.
            centered_term = surprisal - Hq
            
            # We want to INCREASE advantage for Rare items (High Surprisal)
            # Bonus = tau * (Positive for Rare)
            bonus = tau * centered_term
            M= 1+bonus
            # bonus = tau * surprisal
            # Clamp for safety (e.g. max +/- 0.5 change to advantage)
            # bonus = torch.clamp(bonus, -0.5, 0.5)
            print(f"  > [Correct Set] N={corr_mask.sum().item()}")
            print(f"    log p Distribution: {log_pi.detach().cpu().numpy().round(3)}")
            print(f"    q Distribution: {q.detach().cpu().numpy().round(3)}")
            print(f"    Entropy H(q):   {Hq.item():.4f} (Max possible: {np.log(corr_mask.sum().item()):.4f})")
            print(f"    Surprisal (-log q): {surprisal.detach().cpu().numpy().round(2)}")
            print(f"    Centered (S - H):   {centered_term.detach().cpu().numpy().round(2)}")
            print(f"    Final A Bonus:      {bonus.detach().cpu().numpy().round(3)}")
            # advantages[valid_idx] += bonus.detach()
            advantages[valid_idx] *= M.detach()

        # --- PART B: INCORRECT SUBGROUP (Sharpening) ---
        inc_mask = ~corr_mask
        if inc_mask.sum() > 1:
            bad_idx = gidx[inc_mask]
            log_pi = group_logp[inc_mask]
            lens=lengths[inc_mask]
            # 1. Compute q for the error distribution
            log_pi_scaled = log_pi/lens #avg_len
            log_q = log_pi_scaled - torch.logsumexp(log_pi_scaled, dim=0)
            log_q = log_q.detach()
            q = log_q.exp()
            
            # 2. Compute Theoretical Entropy H(q)
            Hq = -(q * log_q).sum()
            # ---- STATS (Incorrect Set) ----
            stats["Hq_incorrect"].append(Hq.detach().cpu())
            stats["num_incorrect"].append(
                torch.tensor(inc_mask.sum().item(), dtype=torch.float32)
            )

            surprisal = -log_q
            # surprisal = -log_pi_scaled            
            # 3. Center by Entropy
            centered_term = surprisal - Hq
            
            # 4. Subtract from Advantage (Penalize Rare Errors)
            penalty = alpha * centered_term
            # penalty = alpha * surprisal
            penalty = torch.clamp(penalty, min=0.0)
            M = 1+penalty
            print(f"  > [Correct Set] N={inc_mask.sum().item()}")
            print(f"    log p Distribution: {log_pi.detach().cpu().numpy().round(3)}")
            print(f"    q Distribution: {q.detach().cpu().numpy().round(3)}")
            print(f"    Entropy H(q):   {Hq.item():.4f} (Max possible: {np.log(corr_mask.sum().item()):.4f})")
            print(f"    Surprisal (-log q): {surprisal.detach().cpu().numpy().round(2)}")
            print(f"    Centered (S - H):   {centered_term.detach().cpu().numpy().round(2)}")
            print(f"    Final A penalty:      {penalty.detach().cpu().numpy().round(3)}")
            # advantages[bad_idx] -= penalty.detach()
            advantages[bad_idx] *= M.detach()
    advantages=advantages.unsqueeze(-1) * response_mask
    return advantages, advantages, stats


def compute_dgrpo_advantage_outcome_A_mod_l(
    token_level_rewards: torch.Tensor,   # [B, T]
    response_mask: torch.Tensor,         # [B, T]
    old_log_prob: torch.Tensor,          # [B, T]
    index,                               # group IDs
    tau: float = 0.1,                    # Diversity strength (Correct)
    alpha: float = 0.1,                  # Sharpening strength (Incorrect)
    epsilon: float = 1e-8,
):
    device = token_level_rewards.device
    stats = {
        "Hq_correct": [],
        "mc_correct": [],
        "num_correct": [],
        "Hq_incorrect": [],
        "num_incorrect": [],
        # ---- NEW: advantage delta tracking ----
        "adv_delta_correct_mean": [],     # mean bonus applied to correct responses
        "adv_delta_correct_max": [],      # max bonus (biggest diversity push)
        "adv_delta_correct_min":[],
        "adv_delta_incorrect_mean": [],   # mean penalty applied to incorrect responses
        "adv_delta_incorrect_max": [],    # max penalty
        "adv_delta_incorrect_min": [],
        "adv_delta_ratio_correct": [],    # mean |bonus| / mean |original_A| for correct
        "adv_delta_ratio_incorrect": [],  # mean |penalty| / mean |original_A| for incorrect
        "adv_original_mean": [],          # mean of original GRPO advantages (sanity check)
        "adv_modified_mean": [],          # mean of modified advantages (sanity check)
    }

    B = token_level_rewards.size(0)
    idx_t = _canon_index(index, batch_size=B, device=device)
    scores = token_level_rewards.sum(dim=-1)                  # [B]
    seq_logp = (old_log_prob * response_mask).sum(dim=-1)     # [B]

    # 2. Compute Standard GRPO Advantage (Baseline)
    advantages = torch.zeros_like(scores)
    unique_groups = torch.unique(idx_t)

    for gid in unique_groups:
        gmask = (idx_t == gid)
        group_scores = scores[gmask]
        if group_scores.numel() > 1:
            mean_g = group_scores.mean()
            std_g = group_scores.std(unbiased=True)
            advantages[gmask] = (group_scores - mean_g) / (std_g + epsilon)
        else:
            advantages[gmask] = group_scores - group_scores.mean()

    # ---- NEW: snapshot original GRPO advantages before modification ----
    original_advantages = advantages.clone()

    # 3. Apply Distributional Corrections
    for gid in unique_groups:
        gidx = torch.nonzero(idx_t == gid, as_tuple=False).squeeze(-1)
        group_logp = seq_logp[gidx]
        group_scores = scores[gidx]
        lengths = (response_mask[gidx] > 0).sum(dim=-1)

        # --- PART A: CORRECT SUBGROUP (Diversity) ---
        corr_mask = (group_scores > 0)
        if corr_mask.sum() > 1:
            valid_idx = gidx[corr_mask]
            log_pi = group_logp[corr_mask]
            lens = lengths[corr_mask]

            log_pi_scaled = log_pi / lens
            log_q = log_pi_scaled - torch.logsumexp(log_pi_scaled, dim=0)
            log_q = log_q.detach()
            q = log_q.exp()

            Hq = -(q * log_q).sum()
            stats["Hq_correct"].append(Hq.detach().cpu())
            stats["mc_correct"].append(q.max().detach().cpu())
            stats["num_correct"].append(
                torch.tensor(corr_mask.sum().item(), dtype=torch.float32)
            )

            surprisal = -log_q
            centered_term = surprisal - Hq
            bonus = tau * centered_term

            # ---- NEW: track the bonus before applying ----
            stats["adv_delta_correct_mean"].append(bonus.mean().detach().cpu())
            stats["adv_delta_correct_max"].append(bonus.max().detach().cpu())
            stats["adv_delta_correct_min"].append(bonus.min().detach().cpu())

            orig_A_correct = original_advantages[valid_idx]
            ratio = bonus.abs().mean() / (orig_A_correct.abs().mean() + 1e-10)
            stats["adv_delta_ratio_correct"].append(ratio.detach().cpu())

            advantages[valid_idx] += bonus.detach()
        elif corr_mask.sum() == 1:
            stats["num_correct"].append(torch.tensor(1.0))

        # --- PART B: INCORRECT SUBGROUP (Sharpening) ---
        inc_mask = ~corr_mask
        if inc_mask.sum() > 1:
            bad_idx = gidx[inc_mask]
            log_pi = group_logp[inc_mask]
            lens = lengths[inc_mask]

            log_pi_scaled = log_pi / lens
            log_q = log_pi_scaled - torch.logsumexp(log_pi_scaled, dim=0)
            log_q = log_q.detach()
            q = log_q.exp()

            Hq = -(q * log_q).sum()
            stats["Hq_incorrect"].append(Hq.detach().cpu())
            stats["num_incorrect"].append(
                torch.tensor(inc_mask.sum().item(), dtype=torch.float32)
            )

            surprisal = -log_q
            centered_term = surprisal - Hq
            penalty = alpha * centered_term
            penalty = torch.clamp(penalty, min=0.0)

            # ---- NEW: track the penalty before applying ----
            stats["adv_delta_incorrect_mean"].append(penalty.mean().detach().cpu())
            stats["adv_delta_incorrect_max"].append(penalty.max().detach().cpu())
            stats["adv_delta_incorrect_min"].append(penalty.min().detach().cpu())

            orig_A_incorrect = original_advantages[bad_idx]
            ratio = penalty.abs().mean() / (orig_A_incorrect.abs().mean() + 1e-10)
            stats["adv_delta_ratio_incorrect"].append(ratio.detach().cpu())

            advantages[bad_idx] -= penalty.detach()

    # ---- NEW: global sanity check stats ----
    stats["adv_original_mean"].append(original_advantages.abs().mean().detach().cpu())
    stats["adv_modified_mean"].append(advantages.abs().mean().detach().cpu())

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages, stats

def compute_dgrpo_advantage_outcome_invq_l(
    token_level_rewards: torch.Tensor,   # [B, T]
    response_mask: torch.Tensor,         # [B, T]
    old_log_prob: torch.Tensor,          # [B, T]
    index,                               # group IDs
    tau: float = 0.3,                    # Interpolation: 0=GRPO, 1=full inverse-q correction
    alpha: float = 0.0,                  # Sharpening strength (Incorrect) — recommend 0
    epsilon: float = 1e-8,
):
    """
    Inverse-q diversity-preserving GRPO advantage.
    
    Instead of adding a bonus to equal advantages, this REPLACES the uniform
    advantage distribution over correct responses with an inverse-probability-weighted
    one. Total positive advantage mass is exactly preserved (zero inflation).
    
    tau interpolates between:
      tau=0: standard GRPO (uniform advantage for all correct)
      tau=1: full inverse-q correction (advantage ∝ 1/q)
    
    The key insight: GRPO gives equal A to all correct responses, but the policy
    gradient reinforces proportional to q (current probability). To get EQUAL
    effective reinforcement, advantages must be proportional to 1/q. tau controls
    how far toward that ideal correction we go.
    """
    device = token_level_rewards.device
    stats = {
        "Hq_correct": [],
        "mc_correct": [],
        "num_correct": [],
        "Hq_incorrect": [],
        "num_incorrect": [],
        # ---- advantage delta tracking ----
        "adv_delta_correct_mean": [],
        "adv_delta_correct_max": [],
        "adv_delta_correct_min": [],
        "adv_delta_incorrect_mean": [],
        "adv_delta_incorrect_max": [],
        "adv_delta_incorrect_min": [],
        "adv_delta_ratio_correct": [],
        "adv_delta_ratio_incorrect": [],
        "adv_original_mean": [],
        "adv_modified_mean": [],
        # ---- inv-q specific stats ----
        "adv_correct_mass_original": [],   # total positive A mass before
        "adv_correct_mass_modified": [],   # total positive A mass after (should match)
        "invq_weight_max": [],             # max weight assigned (monitors blow-up)
        "invq_weight_min": [],             # min weight assigned
    }

    B = token_level_rewards.size(0)
    idx_t = _canon_index(index, batch_size=B, device=device)
    scores = token_level_rewards.sum(dim=-1)                  # [B]
    seq_logp = (old_log_prob * response_mask).sum(dim=-1)     # [B]

    # 2. Compute Standard GRPO Advantage (Baseline)
    advantages = torch.zeros_like(scores)
    unique_groups = torch.unique(idx_t)

    for gid in unique_groups:
        gmask = (idx_t == gid)
        group_scores = scores[gmask]
        if group_scores.numel() > 1:
            mean_g = group_scores.mean()
            std_g = group_scores.std(unbiased=True)
            advantages[gmask] = (group_scores - mean_g) / (std_g + epsilon)
        else:
            advantages[gmask] = group_scores - group_scores.mean()

    # Snapshot original GRPO advantages before modification
    original_advantages = advantages.clone()

    # 3. Apply Inverse-q Redistribution to Correct Set
    for gid in unique_groups:
        gidx = torch.nonzero(idx_t == gid, as_tuple=False).squeeze(-1)
        group_logp = seq_logp[gidx]
        group_scores = scores[gidx]
        lengths = (response_mask[gidx] > 0).sum(dim=-1)

        # --- PART A: CORRECT SUBGROUP (Diversity via Inverse-q) ---
        corr_mask = (group_scores > 0)
        if corr_mask.sum() > 1:
            valid_idx = gidx[corr_mask]
            log_pi = group_logp[corr_mask]
            lens = lengths[corr_mask]
            n_correct = corr_mask.sum().float()

            # Compute q distribution (probability each correct response is "the one" the model favors)
            log_pi_scaled = log_pi / lens
            log_q = log_pi_scaled - torch.logsumexp(log_pi_scaled, dim=0)
            log_q = log_q.detach()
            q = log_q.exp()

            # Stats
            Hq = -(q * log_q).sum()
            stats["Hq_correct"].append(Hq.detach().cpu())
            stats["mc_correct"].append(q.max().detach().cpu())
            stats["num_correct"].append(torch.tensor(n_correct.item(), dtype=torch.float32))

            # --- Inverse-q reweighting ---
            # 1. Compute target distribution: proportional to 1/q
            #    Clamp q to prevent division by zero for extremely rare responses
            q_clamped = q.clamp(min=1e-6)
            inv_q = 1.0 / q_clamped
            target_weights = inv_q / inv_q.sum()   # normalized, sums to 1

            # 2. Uniform weights (what GRPO does: equal advantage for all correct)
            uniform_weights = torch.ones_like(q) / n_correct  # sums to 1

            # 3. Interpolate: tau=0 → GRPO, tau=1 → full inverse-q
            blended_weights = (1.0 - tau) * uniform_weights + tau * target_weights

            # 4. Scale to preserve total advantage mass
            #    GRPO gives each correct response A_base, so total mass = n_correct * A_base
            #    We want: new_A_i = A_base * (blended_weight_i * n_correct)
            #    This ensures sum(new_A_i) = A_base * n_correct = sum(original_A_i)
            A_base = original_advantages[valid_idx[0]]  # all correct have same A in binary reward
            per_response_weights = blended_weights * n_correct  # mean = 1, sum = n_correct

            # 5. Apply: replace advantages for correct set
            new_advantages = A_base * per_response_weights

            # ---- Track deltas ----
            # delta = new_advantages - original_advantages[valid_idx]
            # stats["adv_delta_correct_mean"].append(delta.mean().detach().cpu())
            # stats["adv_delta_correct_max"].append(delta.max().detach().cpu())
            # stats["adv_delta_correct_min"].append(delta.min().detach().cpu())

            # orig_A_mag = original_advantages[valid_idx].abs().mean()
            # if orig_A_mag > 0.01:
            #     stats["adv_delta_ratio_correct"].append(
            #         (delta.abs().mean() / orig_A_mag).detach().cpu()
            #     )

            # stats["adv_correct_mass_original"].append(
            #     original_advantages[valid_idx].sum().detach().cpu()
            # )
            # stats["adv_correct_mass_modified"].append(
            #     new_advantages.sum().detach().cpu()
            # )
            # stats["invq_weight_max"].append(per_response_weights.max().detach().cpu())
            # stats["invq_weight_min"].append(per_response_weights.min().detach().cpu())

            advantages[valid_idx] = new_advantages.detach()

        # elif corr_mask.sum() == 1:
        #     stats["num_correct"].append(torch.tensor(1.0))

        # --- PART B: INCORRECT SUBGROUP (Optional sharpening) ---
        inc_mask = ~corr_mask
        if inc_mask.sum() > 1 and alpha > 0:
            bad_idx = gidx[inc_mask]
            log_pi = group_logp[inc_mask]
            lens = lengths[inc_mask]

            log_pi_scaled = log_pi / lens
            log_q = log_pi_scaled - torch.logsumexp(log_pi_scaled, dim=0)
            log_q = log_q.detach()
            q = log_q.exp()

            Hq = -(q * log_q).sum()
            stats["Hq_incorrect"].append(Hq.detach().cpu())
            stats["num_incorrect"].append(
                torch.tensor(inc_mask.sum().item(), dtype=torch.float32)
            )

            surprisal = -log_q
            centered_term = surprisal - Hq
            penalty = alpha * centered_term
            penalty = torch.clamp(penalty, min=0.0)

            # Track
            # stats["adv_delta_incorrect_mean"].append(penalty.mean().detach().cpu())
            # stats["adv_delta_incorrect_max"].append(penalty.max().detach().cpu())
            # stats["adv_delta_incorrect_min"].append(penalty.min().detach().cpu())

            # orig_A_mag = original_advantages[bad_idx].abs().mean()
            # if orig_A_mag > 0.01:
            #     stats["adv_delta_ratio_incorrect"].append(
            #         (penalty.abs().mean() / orig_A_mag).detach().cpu()
            #     )

            advantages[bad_idx] -= penalty.detach()

        elif inc_mask.sum() > 1:
            # Still track entropy even if alpha=0
            log_pi = group_logp[inc_mask]
            lens = lengths[inc_mask]
            log_pi_scaled = log_pi / lens
            log_q = log_pi_scaled - torch.logsumexp(log_pi_scaled, dim=0)
            q = log_q.exp()
            Hq = -(q * log_q).sum()
            stats["Hq_incorrect"].append(Hq.detach().cpu())
            stats["num_incorrect"].append(
                torch.tensor(inc_mask.sum().item(), dtype=torch.float32)
            )

    # Global sanity check stats
    stats["adv_original_mean"].append(original_advantages.abs().mean().detach().cpu())
    stats["adv_modified_mean"].append(advantages.abs().mean().detach().cpu())

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages, stats

def compute_dgrpo_advantage_outcome_A_diff_smooth(
    token_level_rewards: torch.Tensor,   # [B, T]
    response_mask: torch.Tensor,         # [B, T]
    old_log_prob: torch.Tensor,          # [B, T]
    index,                               # group IDs
    tau: float = 0.1,                    # Diversity strength (Correct)
    alpha: float = 0.1,                  # Sharpening strength (Incorrect)
    epsilon: float = 1e-8,
    diff_smooth_center: str = "none",    # {"mean","lse","none"} (for stability)
):
    """
    Baseline: GRPO advantage per sequence, then broadcast to tokens via response_mask.
    Modification: apply either
      - ours: centered surprisal term (as in your mod version)
      - diff_smooth: directly based on log_pi_scaled (seq logp), optionally centered
    Also logs stats comparing diff_smooth vs ours.
    """
    device = token_level_rewards.device
    stats = {
        # group composition / entropy
        "Hq_correct": [],
        "mc_correct": [],
        "num_correct": [],
        "Hq_incorrect": [],
        "num_incorrect": [],

        # ours deltas
        "adv_delta_correct_mean_ours": [],
        "adv_delta_correct_max_ours": [],
        "adv_delta_correct_min_ours": [],
        "adv_delta_incorrect_mean_ours": [],
        "adv_delta_incorrect_max_ours": [],
        "adv_delta_incorrect_min_ours": [],

        # diff_smooth deltas
        "adv_delta_correct_mean": [],
        "adv_delta_correct_max": [],
        "adv_delta_correct_min": [],
        "adv_delta_incorrect_mean": [],
        "adv_delta_incorrect_max": [],
        "adv_delta_incorrect_min": [],
        "adv_delta_ratio_correct": [],    # mean |bonus| / mean |original_A| for correct
        "adv_delta_ratio_incorrect": [], 

        # diff_smooth - ours comparisons
        "diff_minus_ours_correct_mean": [],
        "diff_minus_ours_correct_max": [],
        "diff_minus_ours_correct_min": [],
        "diff_minus_ours_incorrect_mean": [],
        "diff_minus_ours_incorrect_max": [],
        "diff_minus_ours_incorrect_min": [],

        # advantage sanity
        "adv_original_mean": [],
        "adv_modified_mean": [],
    }

    B = token_level_rewards.size(0)
    idx_t = _canon_index(index, batch_size=B, device=device)
    scores = token_level_rewards.sum(dim=-1)                  # [B]
    seq_logp = (old_log_prob * response_mask).sum(dim=-1)     # [B]  (sequence logp)

    # 1) Standard GRPO advantage on scores (per sequence)
    advantages = torch.zeros_like(scores)
    unique_groups = torch.unique(idx_t)

    for gid in unique_groups:
        gmask = (idx_t == gid)
        group_scores = scores[gmask]
        if group_scores.numel() > 1:
            mean_g = group_scores.mean()
            std_g = group_scores.std(unbiased=True)
            advantages[gmask] = (group_scores - mean_g) / (std_g + epsilon)
        else:
            advantages[gmask] = group_scores - group_scores.mean()

    original_advantages = advantages.clone()

    # helper: center a vector v for diff_smooth
    def _center(v: torch.Tensor) -> torch.Tensor:
        if diff_smooth_center == "mean":
            return v - v.mean()
        if diff_smooth_center == "lse":
            # subtract logsumexp -> like log-softmax up to constant shift
            return v - torch.logsumexp(v, dim=0)
        if diff_smooth_center == "none":
            return v
        raise ValueError(f"Unknown diff_smooth_center={diff_smooth_center}")

    # 2) Apply corrections per group
    for gid in unique_groups:
        gidx = torch.nonzero(idx_t == gid, as_tuple=False).squeeze(-1)
        group_logp = seq_logp[gidx]        # [G]
        group_scores = scores[gidx]        # [G]

        # CORRECT subgroup
        corr_mask = (group_scores > 0)
        if corr_mask.sum() > 1:
            valid_idx = gidx[corr_mask]
            log_pi = group_logp[corr_mask]  # [Gc]

            # ---- OURS: diversity bonus via centered surprisal under q(log_pi) ----
            log_q = log_pi - torch.logsumexp(log_pi, dim=0)
            log_q_det = log_q.detach()
            q = log_q_det.exp()
            Hq = -(q * log_q_det).sum()

            stats["Hq_correct"].append(Hq.detach().cpu())
            stats["mc_correct"].append(q.max().detach().cpu())
            stats["num_correct"].append(torch.tensor(corr_mask.sum().item(), dtype=torch.float32))

            surprisal = -log_q_det
            centered_term = surprisal - Hq
            bonus_ours = tau * centered_term  # [Gc]

            # ---- DIFF_SMOOTH: direct function of log_pi (optionally centered) ----
            log_pi_scaled = log_pi
            bonus_diff = tau * log_pi_scaled  # [Gc]

            # --- log ours stats
            stats["adv_delta_correct_mean_ours"].append(bonus_ours.mean().detach().cpu())
            stats["adv_delta_correct_max_ours"].append(bonus_ours.max().detach().cpu())
            stats["adv_delta_correct_min_ours"].append(bonus_ours.min().detach().cpu())

            # --- log diff stats
            stats["adv_delta_correct_mean"].append(bonus_diff.mean().detach().cpu())
            stats["adv_delta_correct_max"].append(bonus_diff.max().detach().cpu())
            stats["adv_delta_correct_min"].append(bonus_diff.min().detach().cpu())

            # --- compare diff - ours
            delta = (bonus_diff - bonus_ours)
            stats["diff_minus_ours_correct_mean"].append(delta.mean().detach().cpu())
            stats["diff_minus_ours_correct_max"].append(delta.max().detach().cpu())
            stats["diff_minus_ours_correct_min"].append(delta.min().detach().cpu())

            orig_A_correct = original_advantages[valid_idx]
            ratio = bonus_diff.abs().mean() / (orig_A_correct.abs().mean() + 1e-10)
            stats["adv_delta_ratio_correct"].append(ratio.detach().cpu())
            # APPLY (diff_smooth)
            advantages[valid_idx] -= bonus_diff.detach()

        elif corr_mask.sum() == 1:
            stats["num_correct"].append(torch.tensor(1.0))

        # INCORRECT subgroup
        inc_mask = ~corr_mask
        if inc_mask.sum() > 1:
            bad_idx = gidx[inc_mask]
            log_pi = group_logp[inc_mask]  # [Gi]

            # ---- OURS: sharpening penalty via centered surprisal, clamped >=0 ----
            log_q = log_pi - torch.logsumexp(log_pi, dim=0)
            log_q_det = log_q.detach()
            q = log_q_det.exp()
            Hq = -(q * log_q_det).sum()

            stats["Hq_incorrect"].append(Hq.detach().cpu())
            stats["num_incorrect"].append(torch.tensor(inc_mask.sum().item(), dtype=torch.float32))

            surprisal = -log_q_det
            centered_term = surprisal - Hq
            penalty_ours = alpha * centered_term
            # penalty_ours = torch.clamp(penalty_ours, min=0.0)  # [Gi]

            # ---- DIFF_SMOOTH: direct penalty from log_pi (optionally centered) ----
            log_pi_scaled = log_pi
            penalty_diff = alpha * log_pi_scaled  # [Gi]
            # If you want same “only penalize” behavior as ours, clamp:
            # penalty_diff = torch.clamp(penalty_diff, min=0.0)

            # --- log ours stats
            stats["adv_delta_incorrect_mean_ours"].append(penalty_ours.mean().detach().cpu())
            stats["adv_delta_incorrect_max_ours"].append(penalty_ours.max().detach().cpu())
            stats["adv_delta_incorrect_min_ours"].append(penalty_ours.min().detach().cpu())

            # --- log diff stats
            stats["adv_delta_incorrect_mean"].append(penalty_diff.mean().detach().cpu())
            stats["adv_delta_incorrect_max"].append(penalty_diff.max().detach().cpu())
            stats["adv_delta_incorrect_min"].append(penalty_diff.min().detach().cpu())

            # --- compare diff - ours
            delta = (penalty_diff - penalty_ours)
            stats["diff_minus_ours_incorrect_mean"].append(delta.mean().detach().cpu())
            stats["diff_minus_ours_incorrect_max"].append(delta.max().detach().cpu())
            stats["diff_minus_ours_incorrect_min"].append(delta.min().detach().cpu())

            orig_A_incorrect = original_advantages[bad_idx]
            ratio = penalty_diff.abs().mean() / (orig_A_incorrect.abs().mean() + 1e-10)
            stats["adv_delta_ratio_incorrect"].append(ratio.detach().cpu())
            # APPLY (diff_smooth)
            advantages[bad_idx] += penalty_diff.detach()

    stats["adv_original_mean"].append(original_advantages.abs().mean().detach().cpu())
    stats["adv_modified_mean"].append(advantages.abs().mean().detach().cpu())

    # broadcast per-seq advantage to tokens
    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages, stats


def compute_dgrpo_advantage_outcome_A_mod(
    token_level_rewards: torch.Tensor,   # [B, T]
    response_mask: torch.Tensor,         # [B, T]
    old_log_prob: torch.Tensor,          # [B, T]
    index,                               # group IDs
    tau: float = 0.1,                    # Diversity strength (Correct)
    alpha: float = 0.1,                  # Sharpening strength (Incorrect)
    epsilon: float = 1e-8,
):
    device = token_level_rewards.device
    stats = {
        "Hq_correct": [],
        "mc_correct": [],
        "num_correct": [],
        "Hq_incorrect": [],
        "num_incorrect": [],
        # ---- NEW: advantage delta tracking ----
        "adv_delta_correct_mean": [],     # mean bonus applied to correct responses
        "adv_delta_correct_max": [],      # max bonus (biggest diversity push)
        "adv_delta_correct_min": [],
        "adv_delta_incorrect_mean": [],   # mean penalty applied to incorrect responses
        "adv_delta_incorrect_max": [],    # max penalty
        "adv_delta_incorrect_min": [],
        "adv_delta_ratio_correct": [],    # mean |bonus| / mean |original_A| for correct
        "adv_delta_ratio_incorrect": [],  # mean |penalty| / mean |original_A| for incorrect
        "adv_original_mean": [],          # mean of original GRPO advantages (sanity check)
        "adv_modified_mean": [],          # mean of modified advantages (sanity check)
    }

    B = token_level_rewards.size(0)
    idx_t = _canon_index(index, batch_size=B, device=device)
    scores = token_level_rewards.sum(dim=-1)                  # [B]
    seq_logp = (old_log_prob * response_mask).sum(dim=-1)     # [B]

    # 2. Compute Standard GRPO Advantage (Baseline)
    advantages = torch.zeros_like(scores)
    unique_groups = torch.unique(idx_t)

    for gid in unique_groups:
        gmask = (idx_t == gid)
        group_scores = scores[gmask]
        if group_scores.numel() > 1:
            mean_g = group_scores.mean()
            std_g = group_scores.std(unbiased=True)
            advantages[gmask] = (group_scores - mean_g) / (std_g + epsilon)
        else:
            advantages[gmask] = group_scores - group_scores.mean()

    # ---- NEW: snapshot original GRPO advantages before modification ----
    original_advantages = advantages.clone()

    # 3. Apply Distributional Corrections
    for gid in unique_groups:
        gidx = torch.nonzero(idx_t == gid, as_tuple=False).squeeze(-1)
        group_logp = seq_logp[gidx]
        group_scores = scores[gidx]
        lengths = (response_mask[gidx] > 0).sum(dim=-1)

        # --- PART A: CORRECT SUBGROUP (Diversity) ---
        corr_mask = (group_scores > 0)
        if corr_mask.sum() > 1:
            valid_idx = gidx[corr_mask]
            log_pi = group_logp[corr_mask]
            lens = lengths[corr_mask]

            log_pi_scaled = log_pi 
            log_q = log_pi_scaled - torch.logsumexp(log_pi_scaled, dim=0)
            log_q = log_q.detach()
            q = log_q.exp()

            Hq = -(q * log_q).sum()
            stats["Hq_correct"].append(Hq.detach().cpu())
            stats["mc_correct"].append(q.max().detach().cpu())
            stats["num_correct"].append(
                torch.tensor(corr_mask.sum().item(), dtype=torch.float32)
            )

            surprisal = -log_q
            centered_term = surprisal - Hq
            bonus = tau * centered_term

            # ---- NEW: track the bonus before applying ----
            stats["adv_delta_correct_mean"].append(bonus.mean().detach().cpu())
            stats["adv_delta_correct_max"].append(bonus.max().detach().cpu())
            stats["adv_delta_correct_min"].append(bonus.min().detach().cpu())

            orig_A_correct = original_advantages[valid_idx]
            ratio = bonus.abs().mean() / (orig_A_correct.abs().mean() + 1e-10)
            stats["adv_delta_ratio_correct"].append(ratio.detach().cpu())

            advantages[valid_idx] += bonus.detach()
        elif corr_mask.sum() == 1:
            stats["num_correct"].append(torch.tensor(1.0))

        # --- PART B: INCORRECT SUBGROUP (Sharpening) ---
        inc_mask = ~corr_mask
        if inc_mask.sum() > 1:
            bad_idx = gidx[inc_mask]
            log_pi = group_logp[inc_mask]
            lens = lengths[inc_mask]

            log_pi_scaled = log_pi 
            log_q = log_pi_scaled - torch.logsumexp(log_pi_scaled, dim=0)
            log_q = log_q.detach()
            q = log_q.exp()

            Hq = -(q * log_q).sum()
            stats["Hq_incorrect"].append(Hq.detach().cpu())
            stats["num_incorrect"].append(
                torch.tensor(inc_mask.sum().item(), dtype=torch.float32)
            )

            surprisal = -log_q
            centered_term = surprisal - Hq
            penalty = alpha * centered_term
            penalty = torch.clamp(penalty, min=0.0)

            # ---- NEW: track the penalty before applying ----
            stats["adv_delta_incorrect_mean"].append(penalty.mean().detach().cpu())
            stats["adv_delta_incorrect_max"].append(penalty.max().detach().cpu())
            stats["adv_delta_incorrect_min"].append(penalty.min().detach().cpu())

            orig_A_incorrect = original_advantages[bad_idx]
            ratio = penalty.abs().mean() / (orig_A_incorrect.abs().mean() + 1e-10)
            stats["adv_delta_ratio_incorrect"].append(ratio.detach().cpu())

            advantages[bad_idx] -= penalty.detach()

    # ---- NEW: global sanity check stats ----
    stats["adv_original_mean"].append(original_advantages.abs().mean().detach().cpu())
    stats["adv_modified_mean"].append(advantages.abs().mean().detach().cpu())

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages, stats


def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


#change end

def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode="token-mean",
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    # print(
    #     f"ratio stats: mean={ratio.mean().item():.4f}, "
    #     f"max={ratio.max().item():.4f}, min={ratio.min().item():.4f}"
    # )
    # print(
    #     f"pg_clipfrac={pg_clipfrac.item():.6f}, "
    #     f"pg_clipfrac_lower={pg_clipfrac_lower.item():.6f}"
    # )


    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_policy_loss_clip_cov(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    loss_agg_mode="token-mean",
    clip_ratio=0.0002,
    clip_cov_lb=1.0,
    clip_cov_ub=5.0,
):
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.
    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py
    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    assert clip_ratio > 0, "clip_ratio should be larger than 0."
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages- verl_F.masked_mean(advantages, response_mask)) * (log_prob- verl_F.masked_mean(log_prob.detach(), response_mask))
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[:min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr==0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, torch.tensor(0.)


def compute_policy_loss_kl_cov(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    k_ratio=0.0002,
    ppo_kl_coef=1,
):
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.
    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py
    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        k_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    assert k_ratio > 0, "k_ratio should be larger than 0."
    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = - advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = (response_mask > 0)
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0] 
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(k_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * k_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]]

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, torch.tensor(0.), ppo_kl_abs, torch.tensor(0.)



def compute_entropy_loss(logits, response_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=response_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, response_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), response_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
