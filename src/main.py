import ssl

import torch

from data_loader import get_data_loader
from utils import get_analyzer, get_model

ssl._create_default_https_context = ssl._create_unverified_context

# use pydantic create a data data_name class that can be only cifar10 or imagent
# code here

if __name__ == "__main__":
    # config
    data_name = "cifar10"
    batch_size = 512
    input_size = 1000
    pretrained = True
    OOD = False

    train_dataloader, test_dataloader = get_data_loader(data_name, batch_size=batch_size)
    _, input_loader = get_data_loader(data_name, batch_size=input_size)
    model_name = f"resnet34_0"
    weight_path = f"weights/{model_name}.pth"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # use model_name and pretrained to get model
    model = get_model(model_name, pretrained, weight_path)
    model.to(device)

    # dummy input help analyzer to get the shape of output
    if data_name == "cifar10":
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
    else:
        dummy_input = torch.randn(1, 3, 224, 224).to(device)

    input_data = next(iter(input_loader))[0].to(device)
    # prepare data for analyzer

    analyzer = get_analyzer(model, model_name, dummy_input)

    analyzer.save_dimensions()
    analyzer.download_singular_values(input_data)
    # analyzer.download_accuarcy(train_dataloader, test_dataloader, OOD)
