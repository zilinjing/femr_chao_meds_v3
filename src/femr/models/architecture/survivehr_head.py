'''
We use the official survivehr implementation codes from https://github.com/cwlgadd/SurvivEHR 
'''
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class FCNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim, output_dim: int, output_act: nn.Module):
        super().__init__()
        if hidden_dim is None:
            hidden_dims: List[int] = []
        elif isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)

        layers = []
        current_dim = input_dim
        for layer_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, layer_dim))
            layers.append(nn.ReLU())
            current_dim = layer_dim
        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(output_act)
        self.mapping = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mapping(x)


class CondODENet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n: int = 15):
        super().__init__()
        u, w = np.polynomial.legendre.leggauss(n)
        self.register_buffer("u", torch.tensor(u, dtype=torch.float32)[None, :], persistent=False)
        self.register_buffer("w", torch.tensor(w, dtype=torch.float32)[None, :], persistent=False)
        self.n = n
        self.output_dim = output_dim
        self.base_net = FCNet(input_dim, hidden_dim, output_dim, nn.Softplus())

    def ode_mapping(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.base_net(torch.cat((x, t), dim=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tau = torch.matmul(t / 2, 1 + self.u.to(device=t.device, dtype=t.dtype))
        repped_x = x.repeat_interleave(self.n, dim=0)
        dudt = self.ode_mapping(repped_x, tau.reshape(-1, 1))
        f = dudt.reshape(t.shape[0], self.n, self.output_dim)
        integral = (t / 2) * ((self.w.to(device=t.device, dtype=t.dtype).unsqueeze(2) * f).sum(dim=1))
        return 1 - torch.exp(-integral)


class ODESurvMultiple(nn.Module):
    def __init__(self, cov_dim: int, hidden_dim: int, num_risks: int, n: int = 15):
        super().__init__()
        self.K = num_risks
        self.pinet = FCNet(cov_dim, hidden_dim, num_risks, nn.Softmax(dim=1))
        self.odenet = CondODENet(cov_dim + 1, hidden_dim, num_risks, n=n)
        self.last_metrics: Dict[str, torch.Tensor] = {}

    def get_pi(self, x: torch.Tensor) -> torch.Tensor:
        return self.pinet(x)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        pi = self.get_pi(x)
        cdf = self.odenet(x, t)
        return pi * cdf, pi, cdf

    def loss(self, x: torch.Tensor, t: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x.new_tensor(0.0)

        self.last_metrics = {}
        if t.dim() == 1:
            t = t.unsqueeze(1)
        k = k.long()
        eps = 1e-8

        total = x.new_tensor(0.0)

        cens_mask = k == 0
        if cens_mask.any():
            cif_cens, _, _ = self.forward(x[cens_mask], t[cens_mask])
            cens_prob = torch.clamp(1.0 - cif_cens.sum(dim=1), min=eps)
            total = total + torch.log(cens_prob).sum()

        uncens_mask = k > 0
        if uncens_mask.any():
            gathered_index = (k[uncens_mask] - 1).unsqueeze(1)
            _, pi, cdf = self.forward(x[uncens_mask], t[uncens_mask])
            dudt = self.odenet.ode_mapping(x[uncens_mask], t[uncens_mask])

            cdf_sel = cdf.gather(1, gathered_index).squeeze(1)
            dudt_sel = dudt.gather(1, gathered_index).squeeze(1)
            pi_sel = pi.gather(1, gathered_index).squeeze(1)

            log_survival = torch.log(torch.clamp(1.0 - cdf_sel, min=eps))
            log_dudt = torch.log(torch.clamp(dudt_sel, min=eps))
            log_pi = torch.log(torch.clamp(pi_sel, min=eps))
            event_ll = (
                log_survival
                + log_dudt
                + log_pi
            )
            total = total + event_ll.sum()

            log_dudt_detached = log_dudt.detach().float()
            self.last_metrics = {
                "survivehr/log_survival_mean": log_survival.detach().float().mean(),
                "survivehr/log_dudt_mean": log_dudt_detached.mean(),
                "survivehr/log_dudt_p50": torch.quantile(log_dudt_detached, 0.50),
                "survivehr/log_dudt_p95": torch.quantile(log_dudt_detached, 0.95),
                "survivehr/log_dudt_p99": torch.quantile(log_dudt_detached, 0.99),
                "survivehr/log_pi_mean": log_pi.detach().float().mean(),
                "survivehr/event_ll_mean": event_ll.detach().float().mean(),
            }

        return -total


class GaussianValueHeads(nn.Module):
    def __init__(self, hidden_size: int, numerical_risk_ids: List[int], hidden_dim: int = 32):
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                str(risk_id): nn.Sequential(
                    nn.Linear(hidden_size, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 2),
                )
                for risk_id in numerical_risk_ids
            }
        )

    def loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        values: torch.Tensor,
        value_mask: torch.Tensor,
        subject_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if labels.numel() == 0 or len(self.heads) == 0:
            return features.new_tensor(0.0)

        if subject_ids is None or subject_ids.numel() == 0:
            subject_ids = torch.arange(labels.shape[0], device=labels.device, dtype=torch.long)
        else:
            subject_ids = subject_ids.to(device=labels.device, dtype=torch.long)

        unique_subject_ids, subject_inverse = torch.unique(subject_ids, sorted=True, return_inverse=True)
        num_subjects = unique_subject_ids.shape[0]
        if num_subjects == 0:
            return features.new_tensor(0.0)

        default_log_prob = -0.5 * np.log(2.0 * np.pi)
        total = features.new_tensor(0.0)
        for risk_id in torch.unique(labels[labels > 0]).tolist():
            risk_key = str(int(risk_id))
            if risk_key not in self.heads:
                continue
            head = self.heads[risk_key]
            token_mask = labels == int(risk_id)
            if not token_mask.any():
                continue

            token_subjects = subject_inverse[token_mask]
            token_log_prob = features.new_full((int(token_mask.sum().item()),), default_log_prob)
            token_value_mask = value_mask[token_mask]
            if token_value_mask.any():
                params = head(features[token_mask][token_value_mask])
                loc = params[:, 0]
                scale = F.softplus(params[:, 1]) + 1e-4
                dist = torch.distributions.Normal(loc=loc, scale=scale)
                token_log_prob[token_value_mask] = dist.log_prob(values[token_mask][token_value_mask])

            patient_counts = features.new_zeros(num_subjects)
            patient_log_prob = features.new_zeros(num_subjects)
            patient_counts.index_add_(0, token_subjects, features.new_ones(token_subjects.shape[0]))
            patient_log_prob.index_add_(0, token_subjects, token_log_prob)
            token_ll_per_patient = patient_log_prob / (patient_counts + 1e-5)
            total = total + (-token_ll_per_patient.mean())
        return total


class SurvivEHRCausalTaskHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        pretraining_task_info: List[Tuple[str, float]],
        numerical_task_codes: List[str],
        surv_weight: float = 1.0,
        value_weight: float = 0.1,
        survival_hidden_dim: int = 32,
        value_hidden_dim: int = 32,
        quadrature_n: int = 15,
        time_scale_days: float = 1,
        concurrent_strategy: Optional[str] = None,
    ):
        super().__init__()

        task_codes = [item[0] if isinstance(item, (tuple, list)) else item for item in pretraining_task_info]
        self.code_to_index = {code: idx for idx, code in enumerate(task_codes)}
        numerical_risk_ids = [self.code_to_index[code] + 1 for code in numerical_task_codes if code in self.code_to_index]

        total_weight = surv_weight + value_weight
        if total_weight <= 0:
            self.surv_weight = 1.0
            self.value_weight = 0.0
        else:
            self.surv_weight = surv_weight / total_weight
            self.value_weight = value_weight / total_weight

        self.time_scale_days = time_scale_days
        self.concurrent_strategy = concurrent_strategy
        self.survival_model = ODESurvMultiple(
            cov_dim=hidden_size,
            hidden_dim=survival_hidden_dim,
            num_risks=len(task_codes),
            n=quadrature_n,
        )
        self.value_model = GaussianValueHeads(
            hidden_size=hidden_size,
            numerical_risk_ids=numerical_risk_ids,
            hidden_dim=value_hidden_dim,
        )
        self.last_metrics: Dict[str, torch.Tensor] = {}

    def forward(self, features: torch.Tensor, batch: Mapping[str, torch.Tensor], return_logits: bool = False):
        event_time = batch["event_time"].to(features.device, dtype=features.dtype)
        event_label = batch["event_label"].to(features.device, dtype=torch.long)
        event_value = batch["event_value"].to(features.device, dtype=features.dtype)
        value_mask = batch["value_mask"].to(features.device, dtype=torch.bool)
        label_subject_ids = batch.get("label_subject_ids")
        if label_subject_ids is not None:
            label_subject_ids = label_subject_ids.to(features.device, dtype=torch.long)

        if self.time_scale_days > 0:
            event_time = event_time / self.time_scale_days

        if self.concurrent_strategy == "add_noise":
            zero_mask = event_time == 0
            if zero_mask.any():
                exp_dist = torch.distributions.Exponential(rate=1000.0)
                event_time = event_time.clone()
                event_time[zero_mask] = event_time[zero_mask] + exp_dist.sample(event_time[zero_mask].shape).to(
                    event_time.device
                )
        event_time = torch.clamp(event_time, min=1e-6)

        loss_surv = self.survival_model.loss(features, event_time, event_label) / max(event_label.shape[0], 1)
        loss_values = self.value_model.loss(features, event_label, event_value, value_mask, label_subject_ids)
        total_loss = (self.surv_weight * loss_surv) + (self.value_weight * loss_values)

        self.last_metrics = dict(self.survival_model.last_metrics)
        self.last_metrics.update(
            {
                "survivehr/loss_surv": loss_surv.detach(),
                "survivehr/loss_values": loss_values.detach(),
                "survivehr/loss_total": total_loss.detach(),
            }
        )

        result = {}
        if return_logits:
            result["loss_components"] = {
                "loss_surv": loss_surv.detach(),
                "loss_values": loss_values.detach(),
            }
        return total_loss, result
