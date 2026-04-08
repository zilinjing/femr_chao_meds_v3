from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from femr.models.architecture import embedding as base_embedding
from femr.models.architecture.survivehr_head import SurvivEHRCausalTaskHead


class FEMRModel(base_embedding.FEMRModel):
    """Reuse the existing FEMR backbones and swap in a SurvivEHR-style causal head."""

    def create_task_head(self) -> nn.Module:
        if self.loss_type == "survivehr_causal":
            return SurvivEHRCausalTaskHead(self.hidden_size, **self.config.task_config.task_kwargs)
        return super().create_task_head()

    def forward(self, batch: Mapping[str, Any], return_loss=True, return_logits=False, return_reprs=False):
        assert return_loss

        batch = base_embedding.remove_first_dimension(batch)
        input_device = batch["subject_ids"].device
        s = torch.zeros_like(batch["subject_ids"], device=input_device)
        s[1:] = batch["subject_ids"][1:] != batch["subject_ids"][:-1]
        s = torch.cumsum(s, dim=0).type(torch.uint8)

        features = self.backbone(batch["transformer"], s)
        if "task" in batch and self.config.task_config is not None:
            features = features.reshape(-1, features.shape[-1])
            label_indices = batch["transformer"]["label_indices"]
            features = features[label_indices, :]

            task_batch = dict(batch["task"])
            task_batch["label_subject_ids"] = batch["subject_ids"][label_indices]

            loss, result = self.task_model(features, task_batch, return_logits=return_logits)
            if return_reprs:
                result["representations"] = features
            if return_logits or return_reprs:
                result["timestamps"] = batch["transformer"]["timestamps"][label_indices]
                result["subject_ids"] = batch["subject_ids"][label_indices]
            return loss, result

        return super().forward(batch, return_loss=return_loss, return_logits=return_logits, return_reprs=return_reprs)
