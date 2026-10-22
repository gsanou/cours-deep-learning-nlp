"""Exported from notebook pf04_modele_plm.ipynb — edit the notebook source (src/pf04_modele_plm.md), not this file."""

"""Linear-chain CRF with hard transition constraints, batched (forward algorithm + Viterbi)."""
import torch
import torch.nn as nn

FORBIDDEN = -1e4


class ConstrainedCRF(nn.Module):
    def __init__(self, n_tags, allowed=None, start_allowed=None):
        super().__init__()
        self.transitions = nn.Parameter(torch.zeros(n_tags, n_tags))
        self.start = nn.Parameter(torch.zeros(n_tags))
        self.end = nn.Parameter(torch.zeros(n_tags))
        allowed = torch.ones(n_tags, n_tags, dtype=torch.bool) if allowed is None else torch.as_tensor(allowed, dtype=torch.bool)
        start_allowed = torch.ones(n_tags, dtype=torch.bool) if start_allowed is None else torch.as_tensor(start_allowed, dtype=torch.bool)
        self.register_buffer("trans_mask", torch.where(allowed, 0.0, FORBIDDEN))
        self.register_buffer("start_mask", torch.where(start_allowed, 0.0, FORBIDDEN))

    def _trans(self):
        return self.transitions + self.trans_mask

    def neg_log_likelihood(self, emissions, tags, mask):
        """emissions (B, T, K), tags (B, T) long (any valid id on padding), mask (B, T) bool with mask[:, 0] all True."""
        emissions = emissions.float()
        trans, start = self._trans(), self.start + self.start_mask
        B, T, K = emissions.shape
        # score of the gold path
        score = start[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, T):
            step = trans[tags[:, t - 1], tags[:, t]] + emissions[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            score = score + step * mask[:, t]
        last = tags.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
        score = score + self.end[last]
        # log partition function (forward algorithm)
        alpha = start.unsqueeze(0) + emissions[:, 0]
        for t in range(1, T):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + trans.unsqueeze(0), dim=1) + emissions[:, t]
            alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)
        log_z = torch.logsumexp(alpha + self.end.unsqueeze(0), dim=1)
        return (log_z - score).mean()

    @torch.no_grad()
    def decode(self, emissions, mask):
        """Batched Viterbi. Returns a list of tag-id lists (one per sequence, unpadded)."""
        emissions = emissions.float()
        trans, start = self._trans(), self.start + self.start_mask
        B, T, K = emissions.shape
        delta = start.unsqueeze(0) + emissions[:, 0]
        back = []
        for t in range(1, T):
            scores = delta.unsqueeze(2) + trans.unsqueeze(0)              # (B, K_prev, K)
            best, arg = scores.max(dim=1)
            nxt = best + emissions[:, t]
            m = mask[:, t].unsqueeze(1)
            delta = torch.where(m, nxt, delta)
            back.append(torch.where(m, arg, torch.arange(K, device=arg.device).expand(B, K)))  # identity on padding
        delta = delta + self.end.unsqueeze(0)
        best_last = delta.argmax(1)
        paths = []
        lengths = mask.sum(1).tolist()
        for b in range(B):
            tag = int(best_last[b])
            path = [tag]
            for t in range(T - 2, -1, -1):
                tag = int(back[t][b, tag])
                path.append(tag)
            path = path[::-1]
            paths.append(path[: lengths[b]])
        return paths
