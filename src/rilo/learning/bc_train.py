"""Explicit supervised Diffusion Policy training; never runs on import/startup."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from rilo.learning.checkpoints import atomic_json_save, save_dp, save_dp_training, fingerprint
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
    loader_generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(training, batch_size=config.batch_size, shuffle=True, num_workers=0,
                        generator=loader_generator)
    val_loader = DataLoader(validation, batch_size=config.batch_size, shuffle=False, num_workers=0)
    split = {"train": [ep.id for ep in train_episodes], "validation": [ep.id for ep in validation_episodes]}
    best, bad, last, step = float("inf"), 0, {}, 0
    epoch, batches_seen = 0, 0
    saved_checkpoints = []
    checkpoint_index = output / "checkpoints.json"

    def save_training_checkpoint(kind):
        filename = f"checkpoint-step-{step:08d}.pt"
        path = output / filename
        created_at = time.time()
        save_dp_training(model, ema, optimizer, path, step=step,
            metadata={"dataset": selected.manifest(), "split": split, "step": step,
                "weights": "exponential_moving_average", "algorithm": "conditional_action_diffusion_ddim",
                "kind": kind, "created_at": created_at, "resume_supported": False,
                "checkpoint_boundary": "completed_optimizer_update"},
            loader_state={"generator_state": loader_generator.get_state(), "epoch": epoch,
                "batches_seen_in_epoch": batches_seen, "batch_size": config.batch_size,
                "shuffle_permutation_saved": False})
        entry = {"step": step, "file": filename, "kind": kind, "created_at": created_at,
                 "checkpoint_sha256": fingerprint(path), "weights": "exponential_moving_average",
                 "training_state": True, "resume_supported": False}
        saved_checkpoints.append(entry)
        atomic_json_save({"format": "rilo.bc.checkpoints.v1", "checkpoints": saved_checkpoints}, checkpoint_index)
        _emit({**last, "checkpoint_step": step, "checkpoint_file": filename,
               "checkpoint_kind": kind}, callback, output / "metrics.jsonl")

    iterator = iter(loader)
    checkpoint = output/"policy.pt"
    for step in range(1, config.steps+1):
        if control: control()
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            batches_seen = 0
            iterator = iter(loader)
            batch = next(iterator)
        batches_seen += 1
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
        # The optimizer and EMA updates are complete. Save before validation's
        # cooperative cancellation checks, never halfway through an update.
        if step % config.checkpoint_every == 0:
            save_training_checkpoint("periodic")
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
    if not saved_checkpoints or saved_checkpoints[-1]["step"] != step:
        save_training_checkpoint("final")
    metrics = {**last, "best_validation_loss": best, "steps_completed": step,
               "train_episodes": len(train_episodes), "validation_episodes": len(validation_episodes),
               "checkpoint_count": len(saved_checkpoints), "last_checkpoint_step": step}
    return {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": fingerprint(checkpoint),
            "checkpoint_index": str(checkpoint_index.resolve()), "metrics": metrics,
            "config": config.model_dump(mode="json"), "dataset": selected.manifest(), "split": split}
