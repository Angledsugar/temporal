import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# 데이터셋 경로
data_root = Path("dataset/interx_lerobot")

# 메타데이터 읽기
with open(data_root / "meta" / "info.json") as f:
    info = json.load(f)
print("Dataset info:", json.dumps(info, indent=2))

# 에피소드 하나 읽기
table = pq.read_table(data_root / "data" / "chunk-000" / "episode_000000.parquet")
df = table.to_pandas()

actions = np.stack(df["action"].values)      # (T, 7)
states = np.stack(df["observation.state"].values)  # (T, 14)

# --- 0) 모션 텍스트 설명 ---
if "task" in df.columns:
    task_desc = df["task"].iloc[0]
    print(f"Motion description: {task_desc}")
else:
    task_desc = "(no description)"
    # task가 별도 테이블에 있을 수 있음 — episodes.jsonl 확인
    episodes_path = data_root / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        with open(episodes_path) as f:
            ep_meta = json.loads(f.readline())
        task_desc = ep_meta.get("tasks", task_desc)
        print(f"Motion description (from episodes.jsonl): {task_desc}")

# --- 1) 손목 3D 궤적 ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
wrist_xyz = actions[:, :3]  # or states[:, 7:10]
ax.plot(wrist_xyz[:, 0], wrist_xyz[:, 2], wrist_xyz[:, 1])
ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_zlabel('Y')
ax.set_title(f'Wrist 3D Trajectory\n{task_desc}')
plt.show()

# --- 2) Action 시계열 ---
fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
labels = ["x", "y", "z", "qx", "qy", "qz", "gripper"]
for i, (ax, label) in enumerate(zip(axes, labels)):
    ax.plot(actions[:, i])
    ax.set_ylabel(label)
axes[-1].set_xlabel("Timestep")
fig.suptitle(f"Action Dimensions over Time\n{task_desc}")
plt.tight_layout()
plt.show()

# --- 3) Proprioception 시계열 ---
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
groups = [
    ("Joint Angles", slice(0, 7)),
    ("Wrist Position", slice(7, 10)),
    ("Wrist Euler", slice(10, 13)),
    ("Gripper", slice(13, 14)),
]
for ax, (name, s) in zip(axes, groups):
    ax.plot(states[:, s])
    ax.set_ylabel(name)
axes[-1].set_xlabel("Timestep")
fig.suptitle(f"Observation State\n{task_desc}")
plt.tight_layout()
plt.show()