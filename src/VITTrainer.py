
import argparse
from transformers import ViTConfig,ViTModel, ViTForImageClassification


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='ViT Trainer')
    
    parser.add_argument("--dataset")
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--patch_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num_labels", type=int)
    
    args = parser.parse_args()
    
    config = ViTConfig(image_size = args.img_size, 
                       patch_size = args.patch_size, 
                       num_labels = args.num_labels)
    
    vit_model = ViTForImageClassification(config)
    
    
