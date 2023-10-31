import ssl
import time

import torch

from analyzer import get_analyzer
from data_loader import (
    get_balanced_imagenet_input_data,
    get_cifar_input_data,
    get_data_loader,
)
from utils import get_model

ssl._create_default_https_context = ssl._create_unverified_context

# use pydantic create a data data_name class that can be only cifar10 or imagent
# code here

if __name__ == "__main__":
    # config
    model_names = [
        "resnet50",
        "resnet50_swav",
        "convnext",
        "resnet34",
        "resnet18",
    ]
    # _, input_loader = get_data_loader("imagenet", batch_size=15000)
    # input_data = next(iter(input_loader))[0].to("cpu")

    # input_data = get_balanced_imagenet_input_data(15000).to("cpu")
    # print(input_data.shape)
    model_name = "resnet34"

    # data_name = "imagenet"
    data_name = "cifar100"
    batch_size = 512
    input_size = 15000
    pretrained = True
    OOD = True

    train_dataloader, test_dataloader = get_data_loader(data_name, batch_size=batch_size)
    weight_path = f"weights/{model_name}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)

    # use model_name and pretrained to get model

    model = get_model(model_name, data_name, pretrained, weight_path)
    model.to(device)

    # dummy input help analyzer to get the shape of output
    # print("loading data")
    # if data_name == "cifar10":
    #     input_data = get_cifar_input_data().to(device)
    # elif data_name == "imagenet":
    #     input_data = get_balanced_imagenet_input_data(input_size).to(device)

    # print(input_data.shape)

    # Print the results
    # analyzer.save_dimensions()
    # analyzer.save_flowtorch_rank(input_data)
    # analyzer.download_singular_values(input_data)
    # analyzer.download_cov_variances(input_data)
    # analyzer.save_rank(input_data)
    # feature_type = "concat"
    # feature_types = ["concat"]
    for i in range(3):
        feature_type = "avg"

        start = time.time()
        normalize = False
        analyzer = get_analyzer(model, model_name, data_name)
        analyzer.download_accuarcy(train_dataloader, test_dataloader, OOD, feature_type, normalize)
        # analyzer.download_knn_accuracy(
        #     train_dataloader, test_dataloader, OOD, feature_type, normalize
        # )
        end = time.time()
        print(f"total time {feature_type} {'normalize' if normalize else ''}: {end - start}")
