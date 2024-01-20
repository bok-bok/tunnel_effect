import torch

from models.resnet18_large import resnet18 as resnet18_large
from models.resnet18_small import resnet18 as resnet18_small


def get_resnet18_imagenet100(resolution: int):
    checkpoint_path = f"weights/resnet18_imagenet100/{resolution}.pth"
    checkpoint = torch.load(checkpoint_path)
    print(f"loading {checkpoint_path}")

    if resolution in [32, 64]:
        model = resnet18_small(num_classes=100, affine=True)
    else:
        model = resnet18_large(num_classes=100, affine=True)

    # Adjust state_dict keys based on the model type
    state_dict = checkpoint["state_dict"]
    new_state_dict = state_dict

    is_data_parallel = next(iter(model.state_dict())).startswith("module.")
    if is_data_parallel:
        new_state_dict = {
            f"module.{k}" if not k.startswith("module.") else k: v for k, v in state_dict.items()
        }
    else:
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict, strict=True)
    return model
