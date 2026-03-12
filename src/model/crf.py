"""Conditional Random Field (CRF) layer for sequence labeling.

Output Novelty from Context.txt:
A CRF layer replaces the standard Softmax to model transition probabilities
between secondary structure states. This captures physical constraints of
protein folding, preventing impossible sequences (e.g., a single-residue helix).
The CRF learns the 'grammar' of protein folding by calculating the probability
of the entire label sequence jointly.
"""

import torch
import torch.nn as nn


class CRFLayer(nn.Module):
    """Linear-chain Conditional Random Field for sequence labeling.

    Models the joint probability of an output label sequence given emissions
    from the neural network, incorporating learned transition scores between
    consecutive labels.

    Args:
        num_tags: Number of output labels (including PAD).
        pad_idx: Index of the padding label.
    """

    def __init__(self, num_tags, pad_idx=0):
        super().__init__()
        self.num_tags = num_tags
        self.pad_idx = pad_idx

        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

        self._init_constraints()

    def _init_constraints(self):
        """Initialize transition constraints: discourage transitions to/from PAD."""
        with torch.no_grad():
            self.transitions[self.pad_idx, :] = -10000.0
            self.transitions[:, self.pad_idx] = -10000.0
            self.transitions[self.pad_idx, self.pad_idx] = 0.0
            self.start_transitions[self.pad_idx] = -10000.0

    def forward(self, emissions, tags, mask):
        """Compute the negative log-likelihood of the tag sequence.

        Args:
            emissions: Tensor of shape (batch, seq_len, num_tags) with emission scores.
            tags: Tensor of shape (batch, seq_len) with gold tag indices.
            mask: Tensor of shape (batch, seq_len) with 1 for real tokens, 0 for padding.

        Returns:
            Scalar loss (negative log-likelihood averaged over batch).
        """
        mask = mask.float()
        log_numerator = self._compute_score(emissions, tags, mask)
        log_denominator = self._compute_log_partition(emissions, mask)
        nll = log_denominator - log_numerator
        return nll.mean()

    def decode(self, emissions, mask):
        """Find the most likely tag sequence using Viterbi decoding.

        Args:
            emissions: Tensor of shape (batch, seq_len, num_tags).
            mask: Tensor of shape (batch, seq_len).

        Returns:
            List of lists containing the best tag sequence for each example.
        """
        return self._viterbi_decode(emissions, mask)

    def _compute_score(self, emissions, tags, mask):
        """Compute the score of a given tag sequence."""
        batch_size, seq_len, _ = emissions.shape

        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for i in range(1, seq_len):
            trans_score = self.transitions[tags[:, i - 1], tags[:, i]]
            emit_score = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)
            score += (trans_score + emit_score) * mask[:, i]

        last_positions = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, last_positions.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]

        return score

    def _compute_log_partition(self, emissions, mask):
        """Compute the log partition function (forward algorithm)."""
        batch_size, seq_len, num_tags = emissions.shape

        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]

        for i in range(1, seq_len):
            emit_scores = emissions[:, i].unsqueeze(1)
            trans_scores = self.transitions.unsqueeze(0)
            next_alpha = alpha.unsqueeze(2) + trans_scores + emit_scores

            next_alpha = torch.logsumexp(next_alpha, dim=1)
            m = mask[:, i].unsqueeze(1)
            alpha = next_alpha * m + alpha * (1 - m)

        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    def _viterbi_decode(self, emissions, mask):
        """Viterbi decoding for best tag sequences."""
        batch_size, seq_len, num_tags = emissions.shape
        mask = mask.float()

        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        history = []

        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions.unsqueeze(0) + broadcast_emission

            next_score, indices = next_score.max(dim=1)
            m = mask[:, i].unsqueeze(1)
            score = next_score * m + score * (1 - m)
            history.append(indices)

        score += self.end_transitions.unsqueeze(0)

        best_tags_list = []
        seq_ends = mask.long().sum(dim=1) - 1

        _, best_last_tag = score.max(dim=1)

        for b in range(batch_size):
            best_tags = [best_last_tag[b].item()]
            seq_end = seq_ends[b].item()

            for hist in reversed(history[:seq_end]):
                best_tags.append(hist[b, best_tags[-1]].item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list
