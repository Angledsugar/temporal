#!/usr/bin/env python3
"""Deploy TempoRAL pipeline: VLM + Action Expert + MetaController + RL Policy.

Usage:
    python scripts/deploy.py \
        --config temporal/configs/deploy.yaml \
        --vlm gemini-pro \
        --expert checkpoints/phase1/action_expert.pt \
        --meta checkpoints/expert_distill/metacontroller.pt \
        --policy checkpoints/phase3/rl_policy.pt \
        --task "make a cup of coffee"
"""

from __future__ import annotations

import argparse
import logging

import torch
import yaml

from temporal.deploy.capability_prompt import build_capability_prompt
from temporal.deploy.pipeline import DeployConfig, TempoRALPipeline
from temporal.deploy.vlm_interface import VLMConfig, VLMInterface
from temporal.models.action_expert import ActionExpertWrapper
from temporal.models.internal_rl_policy import CausalRLPolicy
from temporal.models.metacontroller import MetaController

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="TempoRAL Deployment")
    parser.add_argument("--config", type=str, default="temporal/configs/deploy.yaml")
    parser.add_argument("--vlm", type=str, default="gemini-pro")
    parser.add_argument("--expert", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, help="Natural language instruction")
    parser.add_argument("--env", type=str, default=None, help="Gym env name (if sim)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    expert = ActionExpertWrapper(width=1024, depth=18, intervention_layer=9).to(device)
    expert.load_state_dict(torch.load(args.expert, map_location=device)["model_state_dict"])
    expert.eval()

    metacontroller = MetaController(n_e=1024, n_z=32).to(device)
    metacontroller.load_state_dict(torch.load(args.meta, map_location=device)["model_state_dict"])
    metacontroller.eval()

    policy = CausalRLPolicy(n_e=1024, n_z=32).to(device)
    policy.load_state_dict(torch.load(args.policy, map_location=device)["model_state_dict"])
    policy.eval()

    # VLM
    vlm_config = VLMConfig(
        provider=config["vlm"]["provider"],
        model=args.vlm,
        temperature=config["vlm"].get("temperature", 0.3),
        max_subtasks=config["vlm"].get("max_subtasks", 20),
    )
    vlm = VLMInterface(vlm_config)

    # Capability prompt
    capability_prompt = build_capability_prompt(
        avg_subtask_duration=config["capability"].get("avg_subtask_duration", 3.0),
        max_objects_per_subtask=config["capability"].get("max_objects_per_subtask", 1),
    )

    # Pipeline
    deploy_config = DeployConfig(
        beta_threshold=config["pipeline"]["beta_threshold"],
        max_steps_per_subtask=config["pipeline"]["max_steps_per_subtask"],
        max_replan_attempts=config["pipeline"].get("max_replan_attempts", 2),
    )
    pipeline = TempoRALPipeline(
        vlm=vlm,
        action_expert=expert,
        metacontroller=metacontroller,
        rl_policy=policy,
        config=deploy_config,
    )

    # Execute
    logger.info(f"Task: {args.task}")
    logger.info(f"Capability prompt:\n{capability_prompt}")

    subtasks = vlm.decompose(args.task, capability_prompt)
    logger.info(f"VLM decomposed into {len(subtasks)} subtasks:")
    for i, st in enumerate(subtasks):
        logger.info(f"  {i+1}. {st}")

    # TODO: Connect to actual environment for execution
    # results = pipeline.execute(args.task, env)
    logger.info("Deployment pipeline ready. Connect environment for execution.")


if __name__ == "__main__":
    main()
