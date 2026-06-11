"""
One-shot migration: split (actor_backbone/critic_backbone + actor/critic heads)
                  -> merged (single actor net, single critic net).

Old keys:  actor_backbone.{0..10}.*   critic_backbone.{0..10}.*
           actor.{0,2,4}.*            critic.{0,2,4}.*   (MLP heads)

New keys:  actor.{0..12}.*  (= old backbone)  +  actor.{13,15,17}.*  (= old head 0,2,4)
           critic.{0..12}.* (= old backbone)  +  critic.{13,15,17}.* (= old head 0,2,4)

The merged Sequential is `*_make_backbone()` (13 modules, indices 0-12) followed by
the head, so the head's Linear layers shift by +13.

Run once:  python migrate_weights.py
"""
import sys
import shutil
import torch
from model import ActorCritic

CHECKPOINT = "weights.pth"
BACKUP     = "weights_pre_merge.pth"

HEAD_SHIFT = 13  # backbone occupies indices 0..12 in the merged net

old_state = torch.load(CHECKPOINT, map_location="cpu")

if not any(k.startswith("actor_backbone.") for k in old_state):
    print("No 'actor_backbone.*' keys found — already merged or wrong file.")
    sys.exit(0)

new_state = {}
for key, value in old_state.items():
    if key.startswith("actor_backbone."):
        new_state["actor." + key[len("actor_backbone."):]] = value
    elif key.startswith("critic_backbone."):
        new_state["critic." + key[len("critic_backbone."):]] = value
    elif key.startswith("actor.") or key.startswith("critic."):
        prefix, idx, param = key.split(".", 2)
        new_state[f"{prefix}.{int(idx) + HEAD_SHIFT}.{param}"] = value
    else:
        new_state[key] = value

# validate against a freshly built model (strict load must succeed)
model = ActorCritic("P1", feature_dim=256)
model.load_state_dict(new_state, strict=True)  # raises if any key/shape mismatches
print("Validation OK — remapped state_dict loads strictly into current model.")

shutil.copy(CHECKPOINT, BACKUP)
print(f"Backup saved to {BACKUP}")

torch.save(new_state, CHECKPOINT)
print(f"Migrated {len(old_state)} keys -> {len(new_state)} keys, saved to {CHECKPOINT}")
