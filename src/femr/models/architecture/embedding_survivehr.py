from __future__ import annotations

from torch import nn

from femr.models.architecture import embedding as base_embedding
from femr.models.architecture.survivehr_head import SurvivEHRCausalTaskHead


class FEMRModel(base_embedding.FEMRModel):
    """Reuse the existing FEMR backbones and swap in a SurvivEHR-style causal head."""

    def create_task_head(self) -> nn.Module:
        if self.loss_type == "survivehr_causal":
            return SurvivEHRCausalTaskHead(self.hidden_size, **self.config.task_config.task_kwargs)
        return super().create_task_head()
