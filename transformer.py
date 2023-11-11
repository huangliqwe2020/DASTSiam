
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import time


def check_inf(tensor):
    return torch.isinf(tensor.detach()).any()


def check_nan(tensor):
    return torch.isnan(tensor.detach()).any()


def check_valid(tensor, type_name):
    if check_inf(tensor):
        print("%s is inf." % type_name)
    if check_nan(tensor):
        print("%s is nan" % type_name)


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, divide_norm=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, divide_norm=divide_norm)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        if num_encoder_layers == 0:
            self.encoder = None
        else:
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, divide_norm=divide_norm)
        decoder_norm = nn.LayerNorm(d_model)
        if num_decoder_layers == 0:
            self.decoder = None
        else:
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                              return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.d_feed = dim_feedforward
        # 2021.1.7 Try dividing norm to avoid NAN
        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5
        self.filter = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, bias=False,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, bias=False,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.cw = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Linear(1, 15),
            nn.ReLU(inplace=True),
            nn.Linear(15, 1),
            nn.Sigmoid(),
        )
        self.filter1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.W = nn.Parameter(torch.ones(2))

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def STforward(self,feat,src,src1, b, c,w1, h1,SC="no",mask=None, pos_embed=False):

        weight = F.softmax(self.W, 0)

        memory = self.encoder(src, src1, src_key_padding_mask=mask, pos=pos_embed)
        feat = feat.permute(1, 2, 0).view(b, c, w1, h1)
        memory = memory.permute(1, 2, 0).view(b, c, w1, h1)
        memory = self.filter(memory)
        if SC=="no":
            channelF = torch.cat([feat, memory], dim=1)
            spatialF = memory + feat
            cw = self.cw(channelF)
            channelF = self.filter1(cw * channelF)
            result = weight[0] * spatialF + weight[1] * channelF
        elif SC=="yes":
            spatialF = memory + feat
            result = spatialF
        else:
            print("error")
            return 0
        result = result.view(b, c, -1).permute(2, 0, 1)

        return result

    def DAforward(self,search,feat,mask=None, pos_embed=False,):
        hs = self.decoder(search, feat, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=None)
        return hs

    def forward(self, search, feat, src, src1, b, c, w1, h1,mode="all", SC="no", mask=None, pos_embed=False,
                return_encoder_output=False):

        if mode=="ST":
            feat = self.STforward(feat,src,src1,b,c,w1,h1,SC)
            return feat
        if mode=="all":
            feat = self.STforward(feat, src, src1, b, c, w1, h1)
            hs = self.DAforward(search,feat)
            return hs,feat
    # def forward(self, query_embed,feat,src,src1, b, c,w1, h1,mask=None, mode="all", pos_embed=False, return_encoder_output=False):
    #
    #     assert mode in ["all", "encoder"]
    #     if self.encoder is None:
    #         memory = src
    #     else:
    #
    #         memory = self.encoder(src,src1, src_key_padding_mask=mask, pos=pos_embed)
    #
    #     memory = memory.permute(1, 2, 0).view(b, c, w1, h1)
    #     a = self.filter(memory)
    #     a = a.view(b, c, -1).permute(2, 0, 1)
    #     feat = a+feat
    #
    #     if mode == "encoder":
    #         return feat
    #     elif mode == "all":
    #         assert len(query_embed.size()) in [2, 3]
    #         if len(query_embed.size()) == 2:
    #             bs = src.size(1)
    #             query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (N,C) --> (N,1,C) --> (N,B,C)
    #         if self.decoder is not None:
    #             tgt = torch.zeros_like(query_embed)
    #             hs = self.decoder(tgt, feat, memory_key_padding_mask=mask,
    #                               pos=pos_embed, query_pos=query_embed)
    #         else:
    #             hs = query_embed.unsqueeze(0)
    #
    #         return hs,feat # (1, B, N, C)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,src1,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                ):
            output = src
            for layer in self.layers:
                output = layer(output,src1, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos=pos)
            if self.norm is not None:
                output = self.norm(output)
            return output



class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before  # first normalization, then add

        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src1,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q,k = self.with_pos_embed(src1, pos),self.with_pos_embed(src1, pos)  # add pos to src
        if self.divide_norm:
            # print("encoder divide by norm")
            q = q / torch.norm(q, dim=-1, keepdim=True) * self.scale_factor
            k = src / torch.norm(k, dim=-1, keepdim=True)
        src2 = self.self_attn(q, k , src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


    def forward(self, src,src1,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        return self.forward_post(src, src1, src_mask, src_key_padding_mask, pos)



class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # self-attention
        q = k = self.with_pos_embed(tgt, query_pos)  # Add object query to the query and key
        if self.divide_norm:
            q = q / torch.norm(q, dim=-1, keepdim=True) * self.scale_factor
            k = k / torch.norm(k, dim=-1, keepdim=True)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # mutual attention
        queries, keys = self.with_pos_embed(tgt, query_pos), self.with_pos_embed(memory, pos)
        if self.divide_norm:
            queries = queries / torch.norm(queries, dim=-1, keepdim=True) * self.scale_factor
            keys = keys / torch.norm(keys, dim=-1, keepdim=True)
        tgt2 = self.multihead_attn(queries, keys, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



def build_transformer(cfg):
    return Transformer(
        d_model=cfg["d_model"],
        dropout=cfg["dropout"],
        nhead=cfg["nhead"],
        dim_feedforward=cfg["dim_feedforward"],
        num_encoder_layers=cfg["num_enclayer"],
        num_decoder_layers=cfg["num_declayer"],
        normalize_before=cfg["nbefore"],
        return_intermediate_dec=False,  # we use false to avoid DDP error,
        divide_norm=cfg["divide_norm"]
    )

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
