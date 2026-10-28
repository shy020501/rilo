"""Explicit supervised Diffusion Policy training; never runs on import/startup."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from rilo.learning.checkpoints import save_dp, fingerprint
from rilo.learning.data import ActionWindowDataset, split_episodes, state_statistics
from rilo.learning.diffusion import DPConfig, DiffusionPolicy, device_for


def _move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def _emit(event, callback, log):
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": time.time(), **event}, allow_nan=False)+"\n")
    if callback:
        callback(event)


def train_bc(snapshot, outdir, config: DPConfig | dict | None = None, *, callback=None, control=None):
    config = DPConfig.model_validate(config or {})
    selected = snapshot.successful() if config.successful_only else snapshot
    if not selected.episodes:
        raise ValueError("BC needs successful task episodes; failed data remain available to residual RL")
    train_episodes, validation_episodes = split_episodes(selected.episodes, config.validation_fraction, config.seed)
    if config.image_size != selected.spec.image_size:
        raise ValueError("Dataset image_size must match DP config.image_size")
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    (output/"dataset.json").write_text(json.dumps(selected.manifest(), indent=2), encoding="utf-8")
    device = device_for(config.device)
    torch.manual_seed(config.seed)
    model = DiffusionPolicy(selected.spec, config).to(device)
    if config.imagenet_init:
        if control: control()
        model.encoder.initialize_imagenet()
    model.encoder.set_state_statistics(*state_statistics(train_episodes, selected.spec))
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": config.encoder_learning_rate},
        {"params": model.denoiser.parameters(), "lr": config.learning_rate}], weight_decay=config.weight_decay)
    training = ActionWindowDataset(train_episodes, selected.spec, config.obs_steps, config.horizon)
    validation = ActionWindowDataset(validation_episodes, selected.spec, config.obs_steps, config.horizon)
    loader = DataLoader(training, batch_size=config.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(config.seed))
    val_loader = DataLoader(validation, batch_size=config.batch_size, shuffle=False, num_workers=0)
    split = {"train": [ep.id for ep in train_episodes], "validation": [ep.id for ep in validation_episodes]}
    best, bad, last, step = float("inf"), 0, {}, 0
    iterator = iter(loader)
    checkpoint = output/"policy.pt"
    for step in range(1, config.steps+1):
        if control: control()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(_move(batch, device))
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite diffusion loss; checkpoint was not promoted")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        with torch.no_grad():
            for moving, current in zip(ema.parameters(), model.parameters()):
                moving.lerp_(current, 1-config.ema_decay)
        last = {"step": step, "train_loss": float(loss.detach().cpu()),
                "parameters": sum(p.numel() for p in model.parameters())}
        if step == 1 or step % config.eval_every == 0 or step == config.steps:
            ema.eval()
            values = []
            generator = torch.Generator(device=device).manual_seed(config.seed+1701)
            with torch.no_grad():
                for index, batch in enumerate(val_loader):
                    if control: control()
                    values.append(float(ema.loss(_move(batch, device), generator=generator).cpu()))
                    if index+1 >= config.validation_batches: break
            score = sum(values)/len(values)
            if not math.isfinite(score):
                raise ValueError("Nonfinite validation loss")
            last["validation_loss"] = score
            if score < best:
                best, bad = score, 0
                save_dp(ema, checkpoint, {"dataset": selected.manifest(), "split": split,
                    "step": step, "validation_loss": best, "parameter_count": last["parameters"],
                    "weights": "exponential_moving_average", "algorithm": "conditional_action_diffusion_ddim"})
            else:
                bad += 1
            _emit(last, callback, output/"metrics.jsonl")
            if step >= config.min_steps and bad >= config.patience:
                break
        elif step % min(config.eval_every, 25) == 0:
            _emit(last, callback, output/"metrics.jsonl")
    metrics = {**last, "best_validation_loss": best, "steps_completed": step,
               "train_episodes": len(train_episodes), "validation_episodes": len(validation_episodes)}
    return {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": fingerprint(checkpoint),
            "metrics": metrics, "config": config.model_dump(mode="json"), "dataset": selected.manifest(), "split": split}
