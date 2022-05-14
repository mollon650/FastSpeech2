import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from model.modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths

from typing import Optional

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len:int,
        mels : Optional[torch.Tensor]=None,
        mel_lens : Optional[torch.Tensor]=None,
        max_mel_len : Optional[int]=None,
        p_targets : Optional[torch.Tensor]=None,
        e_targets : Optional[torch.Tensor]=None,
        d_targets : Optional[torch.Tensor]=None,
        p_control:float=1.0,
        e_control:float=1.0,
        d_control:float=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        # mel_masks = (
        #     get_mask_from_lengths(mel_lens, max_mel_len)
        #     if mel_lens is not None
        #     else None
        # )
        # mel_masks: 
  
        if mel_lens is not None:
            mel_masks = get_mask_from_lengths(mel_lens, max_mel_len)
        else:
            mel_masks = None
            
        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks_,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks_ = self.decoder(output, mel_masks_)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks_,
            src_lens,
            mel_lens,
        )
    # def forward(
    #     self,
    #     speakers,
    #     texts,
    #     src_lens,
    #     max_src_len:int
    # ):
    #     mels: Optional[torch.Tensor]=None
    #     mel_lens: Optional[torch.Tensor]=None
    #     max_mel_len: Optional[int]=None
    #     p_targets: Optional[torch.Tensor]=None
    #     e_targets: Optional[torch.Tensor]=None
    #     d_targets: Optional[torch.Tensor]=None
    #     p_control: float=1.0
    #     e_control: float=1.0
    #     d_control: float=1.0
    #     src_masks = get_mask_from_lengths(src_lens, max_src_len)
    #     # mel_masks = (
    #     #     get_mask_from_lengths(mel_lens, max_mel_len)
    #     #     if mel_lens is not None
    #     #     else None
    #     # )
    #     # mel_masks: 

    #     if mel_lens is not None:
    #         mel_masks = get_mask_from_lengths(mel_lens, max_mel_len)
    #     else:
    #         mel_masks = None
            
    #     output = self.encoder(texts, src_masks)

    #     if self.speaker_emb is not None:
    #         output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
    #             -1, max_src_len, -1
    #         )

    #     (
    #         output,
    #         p_predictions,
    #         e_predictions,
    #         log_d_predictions,
    #         d_rounded,
    #         mel_lens,
    #         mel_masks_,
    #     ) = self.variance_adaptor(
    #         output,
    #         src_masks,
    #         mel_masks,
    #         max_mel_len,
    #         p_targets,
    #         e_targets,
    #         d_targets,
    #         p_control,
    #         e_control,
    #         d_control,
    #     )

    #     output, mel_masks_ = self.decoder(output, mel_masks_)
    #     output = self.mel_linear(output)

    #     postnet_output = self.postnet(output) + output

    #     return (
    #         output,
    #         postnet_output,
    #         p_predictions,
    #         e_predictions,
    #         log_d_predictions,
    #         d_rounded,
    #         src_masks,
    #         mel_masks_,
    #         src_lens,
    #         mel_lens,
    #     )