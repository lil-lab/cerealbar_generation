import os, sys, copy
import pickle
import math
import time
import numpy as np
from typing import Dict, Any, List, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.utils.rnn as rnn_utils  

from agent.environment.position import Position
from agent.environment import card as agent_cards

from . import util
from .map_transformations import pose as pose_lib
from .modules import state_embedder as embedder_lib
from .utilities import initialization
from .helpers import state_representation
from .utilities import hex_util
from .utilities.hex_conv_util import HexConv


def getPositionalEncoding(d_model=768, max_len=1024):
    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
def generate_attention_mask_from_mask_indicies_and_instruction_tensors(feature_attention_mask, instruction_tensors) -> torch.tensor:
    attention_mask = torch.cat([feature_attention_mask, torch.ones(instruction_tensors.shape).to(util.DEVICE).bool()], 1)
    return attention_mask


class CNNLSTMStateEncodingModel(nn.Module):
    def __init__(self, config):
        super(CNNLSTMStateEncodingModel, self).__init__()
        self._d_input = 19
        self._d_embed = config["d_embed"]
        self._d_model = config["d_model"]
        self._embeddings_type = config["embeddings_type"]
        self._breakpoint_type = config["breakpoint_type"]
        if self._breakpoint_type == "":
            pass
        elif self._breakpoint_type == "onehot":
            self._d_input += 1
        else:
            raise ValueError("not supported breakpoint type")

        self._conv = []

        # embedding layer
        self._n_depth = config["n_depth"]

        if self._embeddings_type == "learned":
            if "state_embedder_pretrained_model" in config:
                pretrained_model = config["state_embedder_pretrained_model"]
            else:
                pretrained_model = ""
            self._embedder = embedder_lib.StateEmbedder(
                self._d_embed, pretrained_model)
            self._d_input = self._embedder.embedding_size()
        else:
            if self._embeddings_type == "onehot":
                self._embedder = embedder_lib.StateOnehotEmbedder()
                self._d_input = self._embedder.embedding_size()
            elif self._embeddings_type == "none":
                self._embedder = None
            if self._n_depth != 0:
                conv_module = nn.ModuleList([])
                conv_layer = nn.Conv2d(self._d_input, self._d_model, (1, 1))
                conv_module.append(conv_layer)
                conv_module.append(nn.LeakyReLU())
                if torch.cuda.is_available():
                    conv_module = conv_module.to(util.DEVICE)
                self._conv.append(conv_module)

        # convolutional Layer
        self._rcpf_size = config["rcpf_size"]
        self._cnn_use_norm = config["cnn_use_norm"]
        self._cnn_hex = config["cnn_hex"]
        self._cnn_actv_func = config["cnn_actv_func"]
        padding_size = int((self._rcpf_size-1)/2)

        for d in range(self._n_depth-1):
            conv_module = nn.ModuleList([])
            if d == 0 and self._embeddings_type == "learned":
                conv_in_channels = self._d_input
            else:
                conv_in_channels = self._d_model
            if self._cnn_use_norm:
                norm = nn.InstanceNorm2d(conv_in_channels)
                conv_module.append(norm)
            conv_out_channels: int = self._d_model
            if self._cnn_hex:
                conv_layer = HexConv(conv_in_channels, conv_out_channels,
                                     self._rcpf_size, stride=1, padding=padding_size)
            else:
                conv_layer = nn.Conv2d(conv_in_channels, conv_out_channels,
                                       (self._rcpf_size, self._rcpf_size), padding=(padding_size, padding_size))
            conv_module.append(conv_layer)
            if self._cnn_actv_func == "leaky_relu":
                conv_module.append(nn.LeakyReLU())
            elif self._cnn_actv_func == "tanh":
                conv_module.append(nn.Tanh())
            if torch.cuda.is_available():
                conv_module = conv_module.to(util.DEVICE)
            self._conv.append(conv_module)

        if len(self._conv) == 0:
            self._d_model = self._d_input
        self._conv = nn.ModuleList(self._conv)
        self._conv_output_channel = conv_out_channels

        # feature translation and rotation layers
        self._feature_map_size = config["feature_map_size"] if "feature_map_size" in config else 3
        self._feature_filter_size = config["feature_filter_size"] if "feature_filter_size" in config else self._feature_map_size
        self._rotate_feature_map = config["rotate_feature_map"] if "rotate_feature_map" in config else True
        self._feature_cnn_n_depth = config["feature_cnn_n_depth"] if "feature_cnn_n_depth" in config else 0
        self._feature_merge_type = config["feature_merge_type"] if "feature_merge_type" in config else "sum"
        self._feature_output_dimension = config["feature_output_dimension"] if "feature_output_dimension" in config else 512
        self._feature_cnn_actv_func = config["feature_cnn_actv_func"] if "feature_cnn_actv_func" in config else 0
        self._feature_cnn_use_norm = config["feature_cnn_use_norm"] if "feature_cnn_use_norm" in config else True
        self._feature_conv = []
        try:
            assert(self._feature_output_dimension * (self._feature_map_size)**2 //
                   (self._feature_map_size)**2 == self._feature_output_dimension)
        except:
            raise ValueError(
                "Feature output dimension is not divisible by the nubmer of hexes to be clopped.")

        for d in range(self._feature_cnn_n_depth):
            conv_module = nn.ModuleList([])
            if self._feature_cnn_use_norm:
                norm = nn.InstanceNorm2d(512) #! not adaptive  
            conv_module.append(norm)

            if self._feature_merge_type == "cat":
                traj_output_channel = self._feature_output_dimension // (self._feature_map_size)**2
                padding = (self._feature_filter_size-1)//2
                if self._cnn_hex:
                    conv_layer = HexConv(self._conv_output_channel, traj_output_channel,
                                         self._feature_filter_size, stride=1, padding=padding)
                else:
                    conv_layer = nn.Conv2d(self._conv_output_channel, traj_output_channel, (
                        self._feature_filter_size, self._feature_filter_size), padding=(padding, padding))
                    self._conv_output_channel = traj_output_channel
            elif self._feature_merge_type == "sum":
                traj_output_channel = self._conv_output_channel
                if self._cnn_hex:
                    conv_layer = HexConv(self._conv_output_channel, traj_output_channel,
                                         self._feature_map_size, stride=1, padding=0)
                else:
                    conv_layer = nn.Conv2d(self._conv_output_channel, traj_output_channel,
                                           (self._feature_map_size, self._feature_map_size), padding=(0, 0))

            conv_module.append(conv_layer)

            if self._cnn_actv_func == "tanh":
                conv_module.append(nn.Tanh())
            self._feature_conv.append(conv_module)

        self._feature_conv = nn.ModuleList(self._feature_conv)
        if self._feature_merge_type == "cat":
            self._conv_output_channel = self._feature_output_dimension
            self._d_model = self._feature_output_dimension
        elif self._feature_merge_type == "sum":
            self._d_model = traj_output_channel
        self._rotator = hex_util.Hex_Rotator()

        # LSTM Layer
        # 0. Pose + breakpoint embedder
        # 1. Preprocessing linear layer (optional)
        # 2. LSTM layer
        #    2.1 Optional skip connection
        self._lstm_input_merge_type = config["lstm_input_merge_type"]
        self._lstm_output_merge_type = config["lstm_output_merge_type"]
        self._lstm_skip = config["lstm_skip"]

        if self._lstm_input_merge_type == "cat":
            self._traj_break_embedder = embedder_lib.TrajBreakEmbedder(config["lstm_pb_dim"])
            lstm_input_dim = self._d_model + config["lstm_pb_dim"]
            lstm_output_dim = config["lstm_d_model"]
        elif self._lstm_input_merge_type == "add":
            self._traj_break_embedder = embedder_lib.TrajBreakEmbedder(self._d_model)
            lstm_input_dim = self._d_model
            lstm_output_dim = config["lstm_d_model"]

        self._lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_output_dim,
            num_layers=config["lstm_num_layers"],
            bidirectional=config["lstm_bidirectional"],
            dropout=config["lstm_dropout"],
            batch_first=True,
        )

        if config["lstm_bidirectional"]:
            lstm_output_dim = lstm_output_dim * 2
        else:
            lstm_output_dim = config["lstm_d_model"]

        if self._lstm_skip:
            if self._lstm_output_merge_type == "spatial-cat":
                self._d_model = lstm_output_dim + self._d_model // (self._feature_map_size)**2
        else:
            try:
                assert(self._lstm_output_merge_type != "spatial-cat")
            except:
                raise ValueError(
                    "Spaitial conceteneation option is only supported for LSTM with a skip coonection.")
            self._d_model = lstm_output_dim

        if torch.cuda.is_available():
            self._lstm.to(util.DEVICE)

    def forward(self, x, traj=None, bkpoint=None):
        input = x.transpose(1, 3)  # [BWHC] ==> [BCHW]
        input = input.transpose(2, 3)  # [BCHW] ==>[BCWH]

        # input processing
        input[:, 15, :, :] = torch.clamp(input[:, 15, :, :], 0, 1)
        input = input.detach()
        input = input.contiguous()

        # embeddings layer
        if self._embedder is not None:
            input = self._embedder(input)

        # hex CNN 1
        conv_outputs: List[torch.Tensor] = list()
        for i, layer in enumerate(self._conv):
            conv_in = input if i == 0 else conv_outputs[-1]
            x = conv_in
            for l in layer:
                x = l(x)
            # residual coneection (if k != 1)
            if (i != 0 and i != self._n_depth):
                x = x + conv_outputs[-1]
            conv_outputs.append(x)

        if len(self._conv) == 0:
            final_feature = input
        else:
            final_feature = conv_outputs[-1]

        # cropping features
        if self._feature_map_size != 1:
            center = (self._feature_map_size-1) // 2
            # Syntax: https://discuss.pytorch.org/t/is-there-a-way-to-pad-a-tensor-instead-of-variable/10448/2
            final_feature = F.pad(final_feature, (center, center, center, center))

        features = []
        spatial_features = []
        pb_features = []

        batch_idx_list = [[i for _ in range(len(t))] for i, t in enumerate(traj)]
        final_feature_mask_indicies = [len(t) for t in traj]
        batch_idx = []
        for l in batch_idx_list:
            batch_idx += l
        batch_idx = torch.tensor(batch_idx).to(util.DEVICE)
        coords = torch.cat(traj,0)
        h_mask = coords[:, 0]
        w_mask = coords[:, 1]
        pose = coords[:, 2]
        h_mask = h_mask.detach()
        w_mask = w_mask.detach()

        if self._feature_map_size == 1:
            feature = final_feature[i, :, h_mask, w_mask]
            feature = feature.permute(1, 0)
        else:
            rows = [h_mask + (slack-center) for slack in range(self._feature_map_size)]
            rows = torch.stack(rows, 0).unsqueeze(1)
            rows = rows.repeat(1, self._feature_map_size, 1)
            rows = rows + center  # need to add center bc of padding
            rows = rows.detach()
            cols = [w_mask + (slack-center) for slack in range(self._feature_map_size)]
            cols = torch.stack(cols, 0).unsqueeze(0)
            cols = cols.repeat(self._feature_map_size, 1, 1)
            cols = cols + center  # need to add center bc of padding
            cols = cols.detach()
            batch_idx = batch_idx.unsqueeze(0).unsqueeze(0)
            batch_idx = batch_idx.repeat(self._feature_map_size, self._feature_map_size, 1)
            feature = final_feature[batch_idx, :, rows, cols]
            feature = feature.permute(2, 3, 0, 1) # TxDxHxW

        # rotate features
        if self._rotate_feature_map:
            mask_l = len(h_mask)
            # converting to offset coordinates
            pose_position = torch.tensor([[center+center//2, center]
                                            for _ in range(mask_l)]).to(util.DEVICE)
            pose_rot = (pose-1) * math.radians(60)
            pose_obj = pose_lib.Pose(pose_position, pose_rot)
            new_feature = self._rotator.translate_and_rotate(feature, pose_obj)
            feature = new_feature

        # hex CNN 2
        feature = feature.contiguous()
        x = feature
        for i, layer in enumerate(self._feature_conv):
            for l in layer:
                x = l(x)

        spatial_feature = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]) #LxDX(H*W)
        feature = torch.cat([spatial_feature[:, :, i]
                                for i in range(spatial_feature.shape[2])], 1) # LxDX(H*W)

        # attach pose features
        bk_onehot = torch.zeros(pose.shape).long().to(util.DEVICE)
        pose_bk_raw_features = torch.stack([pose, bk_onehot], 0)
        pb_feature = self._traj_break_embedder(pose_bk_raw_features)
        if self._lstm_input_merge_type == "cat":
            feature = torch.cat([feature, pb_feature], 1)
        elif self._lstm_input_merge_type == "add":
            feature += pb_feature

        spatial_features = torch.split(spatial_feature, final_feature_mask_indicies)
        features = torch.split(feature, final_feature_mask_indicies)

        # LSTM layer
        # reference: https://discuss.pytorch.org/t/how-can-i-compute-seq2seq-loss-using-mask/861
        lstm_input = pad_sequence(features, 1, padding_value=0)
        unpacked = lstm_input.permute(1, 0, 2)
        packed = rnn_utils.pack_padded_sequence(unpacked, final_feature_mask_indicies, enforce_sorted=False)
        outputs, _ = self._lstm(packed, None)
        unpacked, unpacked_len = rnn_utils.pad_packed_sequence(outputs)
        final_feature = unpacked.permute(1, 0, 2)
        final_feature = final_feature.contiguous()

        if self._lstm_skip:
            spatial_features = pad_sequence(spatial_features, 1, padding_value=0)
            final_feature = final_feature.unsqueeze(-1)
            final_feature = final_feature.repeat(1, 1, 1, spatial_features.shape[-1])
            final_feature = torch.cat([final_feature, spatial_features], 2)
            final_feature = final_feature.permute(0, 1, 3, 2)
            final_feature = final_feature.contiguous().view(
                (final_feature.shape[0], final_feature.shape[1]*final_feature.shape[2], final_feature.shape[3]))

        final_feature = final_feature.contiguous()
  
        # generate attention mask for feature
        feature_attention_mask = torch.ones(final_feature.shape[:2]).to(util.DEVICE)
        batch_size = final_feature.shape[0]
        neighbor_size = spatial_features.shape[-1]
        for i in range(batch_size):
            feature_attention_mask[i, neighbor_size*final_feature_mask_indicies[i]:] = 0
        feature_attention_mask = feature_attention_mask.bool()

        return final_feature, feature_attention_mask 

    def get_dimension(self):
        return self._d_model


