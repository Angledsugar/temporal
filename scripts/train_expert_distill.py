#!/usr/bin/env python3
"""Expert Distill: Train MetaController on frozen action expert's residual stream.

Discovers temporal abstractions (subtask boundaries) from the frozen
Phase 1 action expert, without any boundary supervision.

Usage:
    # HHI-specialized (Inter-X)
    python scripts/train_expert_distill.py \
        --config configs/expert_distill_interx.yaml

    # Quick test
    python scripts/train_expert_distill.py \
        --expert checkpoints/phase1_interx/action_expert_final.pt \
        --data dataset/interx_npz \
        --max-steps 500 --batch-size 8

    # With DROID data (original pipeline)
    python scripts/train_expert_distill.py \
        --config configs/expert_distill.yaml

    # Evaluate existing checkpoint
    python scripts/train_expert_distill.py \
        --config configs/expert_distill_interx.yaml \
        --evaluate --resume checkpoints/expert_distill_interx/metacontroller_final.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Expert Distill: MetaController training")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--expert", type=str, default=None, help="Action expert checkpoint (overrides config)")
    parser.add_argument("--data", type=str, default=None, help="Dataset path (overrides config)")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from MetaController checkpoint")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--alpha", type=float, default=None, help="Override KL weight")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after training")
    args = parser.parse_args()

    # Load config
    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Load frozen action expert
    from temporal.models.action_expert import ActionExpert, ActionExpertConfig

    expert_path = args.expert or config.get("expert_checkpoint")
    model_cfg = config.get("model", {})

    expert_config = ActionExpertConfig(
        width=model_cfg.get("n_e", 1024),
        depth=18,
        mlp_dim=4096,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        controlled_layer=model_cfg.get("controlled_layer", 9),
        action_dim=7,
    )

    expert = ActionExpert(expert_config).to(device)

    if expert_path and Path(expert_path).exists():
        ckpt = torch.load(expert_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        expert.load_state_dict(state_dict)
        logger.info(f"Loaded frozen expert from {expert_path}")
    else:
        logger.warning(
            "No expert checkpoint loaded — using random initialization. "
            "This is only useful for testing the pipeline."
        )

    expert.eval()
    for p in expert.parameters():
        p.requires_grad = False

    n_expert = sum(p.numel() for p in expert.parameters())
    logger.info(f"ActionExpert: {n_expert / 1e6:.1f}M parameters (frozen)")

    # Build MetaController
    from temporal.models.metacontroller import MetaController

    meta = MetaController(
        n_e=model_cfg.get("n_e", 1024),
        n_z=model_cfg.get("n_z", 32),
        rank=model_cfg.get("rank", 32),
        encoder_hidden=model_cfg.get("encoder_hidden", 128),
    ).to(device)

    n_meta = sum(p.numel() for p in meta.parameters())
    logger.info(f"MetaController: {n_meta / 1e6:.2f}M parameters")

    # Resume MetaController if specified
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        meta_sd = ckpt.get("metacontroller_state_dict", ckpt)
        meta.load_state_dict(meta_sd)
        logger.info(f"Resumed MetaController from {args.resume}")

    # Data
    data_cfg = config.get("data", {})
    data_root = args.data or config.get("data_root")
    if not data_root:
        parser.error("Must provide --data or config with data_root")

    from temporal.data.human_motion_dataset import create_dataloader

    batch_size = args.batch_size or data_cfg.get("batch_size", 32)
    dataloader = create_dataloader(
        data_root=data_root,
        split=data_cfg.get("split", "train"),
        batch_size=batch_size,
        num_workers=data_cfg.get("num_workers", 4),
        max_length=data_cfg.get("max_length", 256),
    )

    # Build trainer config
    from temporal.training.expert_distill_trainer import ExpertDistillConfig, ExpertDistillTrainer

    training_cfg = config.get("training", {})
    distill_config = ExpertDistillConfig(
        alpha=args.alpha or training_cfg.get("alpha", 0.05),
        learning_rate=training_cfg.get("learning_rate", 1e-3),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        max_steps=args.max_steps or training_cfg.get("max_steps", 200_000),
        grad_clip=training_cfg.get("grad_clip", 1.0),
        controlled_layer=model_cfg.get("controlled_layer", 9),
        batch_size=batch_size,
        log_every=training_cfg.get("log_every", 100),
        save_every=training_cfg.get("save_every", 5_000),
        output_dir=args.output or config.get("output_dir", "checkpoints/expert_distill/"),
    )

    Path(distill_config.output_dir).mkdir(parents=True, exist_ok=True)

    # Train
    trainer = ExpertDistillTrainer(
        action_expert=expert,
        metacontroller=meta,
        dataloader=dataloader,
        config=distill_config,
    )

    logger.info("=" * 60)
    logger.info("Starting Expert Distill training")
    logger.info(f"  alpha={distill_config.alpha}, lr={distill_config.learning_rate}")
    logger.info(f"  max_steps={distill_config.max_steps}, batch_size={batch_size}")
    logger.info(f"  output_dir={distill_config.output_dir}")
    logger.info("=" * 60)

    history = trainer.train()

    # Save final checkpoint
    final_path = Path(distill_config.output_dir) / "metacontroller_final.pt"
    torch.save({
        "metacontroller_state_dict": meta.state_dict(),
        "action_decoder_state_dict": trainer.action_decoder.state_dict(),
        "history": history,
        "config": vars(distill_config),
    }, final_path)
    logger.info(f"Saved final checkpoint: {final_path}")

    # Evaluate
    if args.evaluate:
        logger.info("=" * 60)
        logger.info("Evaluating boundary detection quality")
        logger.info("=" * 60)
        eval_metrics = trainer.evaluate_boundaries(num_samples=200)
        logger.info(f"Evaluation results: {eval_metrics}")

    logger.info("Expert Distill complete.")


if __name__ == "__main__":
    main()
