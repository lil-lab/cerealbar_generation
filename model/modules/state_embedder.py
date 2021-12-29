""" Environment embedder for static parts of an environment (e.g., terrain, static props).

Classes:
    StaticEnvironmentEmbedder (nn.Module): Embeds the static information about an environment.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from IPython import embed

import torch
import torch.nn as nn

from ..helpers import state_representation
from .. import util

if TYPE_CHECKING:
    from typing import List, Optional


class StateOnehotEmbedder(nn.Module):
    def __init__(self):
        super(StateOnehotEmbedder, self).__init__()
        self._state_rep = state_representation.StateRepresentation()
        self._embedder = nn.Embedding(self._state_rep.get_state_length(), self._state_rep.get_state_length())
        self._embedder.weight = nn.Parameter(torch.eye(self._state_rep.get_state_length(), self._state_rep.get_state_length()))
        self._embedder.weight.requires_grad = False
        self._get_prefix_tensor()
        self._embedder.weight[self._prefix_tensor, self._prefix_tensor] = 0
        if torch.cuda.is_available():
            self._embedder = self._embedder.to(util.DEVICE)
        self._prefix_tensor = self._prefix_tensor.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    def _get_prefix_tensor(self):
        self._prefix_tensor = torch.zeros((self._state_rep.get_state_num_channel()))
        self._prefix_tensor[0] = 0
        self._prefix_tensor[1] = self._prefix_tensor[0] + len(self._state_rep.get_terrains())
        self._prefix_tensor[2] = self._prefix_tensor[1] + len(self._state_rep.get_prop_types())
        self._prefix_tensor[3] = self._prefix_tensor[2] + len(self._state_rep.get_tree_types())
        self._prefix_tensor[4] = self._prefix_tensor[3] + len(self._state_rep.get_hut_color())
        self._prefix_tensor[5] = self._prefix_tensor[4] + len(self._state_rep.get_hut_rotation())
        self._prefix_tensor[6] = self._prefix_tensor[5] + len(self._state_rep.get_plant_types())
        self._prefix_tensor[7] = self._prefix_tensor[6] + len(self._state_rep.get_windmill_rotation())
        self._prefix_tensor[8] = self._prefix_tensor[7] + len(self._state_rep.get_tower_rotation())
        self._prefix_tensor[9] = self._prefix_tensor[8] + len(self._state_rep.get_tent_rotation())
        self._prefix_tensor[10] = self._prefix_tensor[9] + len(self._state_rep.get_card_color())
        self._prefix_tensor[11] = self._prefix_tensor[10] + len(self._state_rep.get_card_shape())
        self._prefix_tensor[12] = self._prefix_tensor[11] + len(self._state_rep.get_card_count())
        self._prefix_tensor[13] = self._prefix_tensor[12] + len(self._state_rep.get_card_selection())
        self._prefix_tensor[14] = self._prefix_tensor[13] + len(self._state_rep.get_leader_rotation())
        self._prefix_tensor[15] = self._prefix_tensor[14] + len(self._state_rep.get_follower_rotation())
        self._prefix_tensor[16] = self._prefix_tensor[15] + len(self._state_rep.get_trajectory())
        self._prefix_tensor[17] = self._prefix_tensor[16] + len(self._state_rep.get_goals())
        self._prefix_tensor[18] = self._prefix_tensor[17] + len(self._state_rep.get_avoidance())

        self._prefix_tensor = self._prefix_tensor
        if torch.cuda.is_available():
            self._prefix_tensor = self._prefix_tensor.to(util.DEVICE)
        self._prefix_tensor = self._prefix_tensor.detach()

    def embedding_size(self) -> int:
        """ Returns the embedding size for a tensor coming out of the forward method. """
        return self._embedder.weight.shape[1]

    def forward(self, state) -> torch.Tensor:
        # [1] Update the indices of each tensor to come after the previous tensors
        state_properties = state + self._prefix_tensor
        state_properties = state_properties.long()
        state_properties = state_properties.detach()
        # [3] Embed.
        # The output should be of size N x B x H x W x C.
        #   N is the number of property types.
        #   B is the batch size.
        #   H is the height of the environment.
        #   W is the width of the environment.
        #   C is the embedding size.
        all_property_embeddings = self._embedder(state_properties)

        # Permute so it is B x N x C x H x W.
        all_property_embeddings = all_property_embeddings.permute(0, 4, 2, 3, 1)

        # Then sum across the property type dimension.
        emb_state = torch.sum(all_property_embeddings, dim=4)
        return emb_state


class StateEmbedder(StateOnehotEmbedder):
    """ Embedder for the static parts of an environment.

    Args:
        state_rep: DenseStateRepresentation. Formal representation of the environment properties.
        embedding_size: int. The size of each underlying embedding.
        zero_out: bool. Whether or not to zero-out embeddings which represent absence of a property.
            The terrain embedder is never zeroed out.
    """

    def __init__(self, embedding_dim, pretrained_model=""):
        super(StateOnehotEmbedder, self).__init__()
        self._state_rep = state_representation.StateRepresentation()
        self._embedder = nn.Embedding(self._state_rep.get_state_length(), embedding_dim)
        torch.nn.init.xavier_normal_(self._embedder.weight)
        self._get_prefix_tensor()
        if pretrained_model != "":
            weight = torch.load(pretrained_model)["_embedder.weight"].cpu() # TODO fix this. Model saving in embedder has to be fixed first.
            self._embedder.weight.data.copy_(weight)
        else:
            weight = self._embedder.weight.detach()
            weight[self._prefix_tensor.long(), :] = 0
            self._embedder.weight.data.copy_(weight)
        if torch.cuda.is_available():
            self._embedder = self._embedder.to(util.DEVICE)
        self._prefix_tensor = nn.Parameter(self._prefix_tensor.unsqueeze(0).unsqueeze(2).unsqueeze(3))
        self._prefix_tensor.requires_grad = False

class TrajBreakEmbedder(StateEmbedder):
    """ Embedder for the static parts of an environment.

    Args:
        state_rep: DenseStateRepresentation. Formal representation of the environment properties.
        embedding_size: int. The size of each underlying embedding.
        zero_out: bool. Whether or not to zero-out embeddings which represent absence of a property.
            The terrain embedder is never zeroed out.
    """

    def __init__(self, embedding_dim):
        super(StateOnehotEmbedder, self).__init__()
        self._state_rep = state_representation.StateRepresentation()
        self._embedder = nn.Embedding(9, embedding_dim)
        torch.nn.init.xavier_normal_(self._embedder.weight)
        self._get_prefix_tensor()
        weight = self._embedder.weight.detach()
        weight[self._prefix_tensor.long(), :] = 0
        self._embedder.weight.data.copy_(weight)
        if torch.cuda.is_available():
            self._embedder = self._embedder.to(util.DEVICE)
        self._prefix_tensor = nn.Parameter(self._prefix_tensor.unsqueeze(1))
        self._prefix_tensor.requires_grad = False

    def _get_prefix_tensor(self):
        self._prefix_tensor = torch.zeros((2))
        self._prefix_tensor[0] = 0
        self._prefix_tensor[1] = self._prefix_tensor[0] + len(self._state_rep.get_follower_rotation())
        self._prefix_tensor = self._prefix_tensor

        if torch.cuda.is_available():
            self._prefix_tensor = self._prefix_tensor.to(util.DEVICE)
        self._prefix_tensor = self._prefix_tensor.detach()

    def forward(self, state) -> torch.Tensor:
        # Update the indices of each tensor to come after the previous tensors
        state_properties = state + self._prefix_tensor
        state_properties = state_properties.long()
        state_properties = state_properties.detach()
        all_property_embeddings = self._embedder(state_properties)

        # Then sum across the property type dimension.
        emb_state = torch.sum(all_property_embeddings, dim=0)
        return emb_state
