import torch
import numpy as np

def inspect_model(filepath):
    try:
        checkpoint = torch.load(filepath, map_location='cpu')
        print(f"File: {filepath}")
        if isinstance(checkpoint, dict):
            print("Contents of dict:", checkpoint.keys())
            if 'epsilon' in checkpoint:
                print(f"Epsilon: {checkpoint['epsilon']}")
            if 'network_state_dict' in checkpoint:
                sd = checkpoint['network_state_dict']
                print("State dict keys:", sd.keys())
                for k, v in sd.items():
                    print(f"  {k}: shape {v.shape}, mean {v.mean().item():.4f}, std {v.std().item():.4f}, max {v.max().item():.4f}, min {v.min().item():.4f}")
        else:
            print("Saved object is not a dict. It might be a direct state_dict or a full model.")
            if hasattr(checkpoint, 'keys'):
                print("Keys:", checkpoint.keys())
            else:
                print("Loaded object:", type(checkpoint))
    except Exception as e:
        print(f"Error loading {filepath}: {e}")

if __name__ == "__main__":
    inspect_model("model_0.pth")
    print("-" * 20)
    inspect_model("model_1.pth")
    print("-" * 20)
    inspect_model("post_trening_0.pth")
    print("-" * 20)
    inspect_model("stary_model.pth")
