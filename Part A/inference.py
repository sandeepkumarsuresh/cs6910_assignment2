import torch
from preprocess import Custom_val_dataset
from torch.utils.data import DataLoader
from CNN import CNN
import matplotlib.pyplot as plt
import wandb
import numpy as np

def calculate_accuracy(model,dataloader):

    model.eval() # Turns of Specific Parameters like BatchNorm , Dropout

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images , labels in (dataloader):
            images,labels = images.to(device),labels.to(device)

            predictions = torch.argmax(model(images),dim=1)
            labels = torch.squeeze(labels)

            correct_prediction = sum(predictions==labels).item()

            total_correct += correct_prediction
            total_samples += len(images)

    return round(total_correct/total_samples , 3)

if __name__ == '__main__':

    # wandb.login()

    # run = wandb.init(
    #     project ="dl_ass2"
    # )  

    BATCH_SIZE = 32
    MODEL_PATH = 'model_dir/Adam Optimizer:batch_size:32filters:128dropout:0.3filter_multiplier:0.5kernel_size:3'
    class_map = (
                "Fungi",
                "Insecta",
                "Animalia",
                "Arachnida",
                "Aves",
                "Mollusca",
                "Reptilia",
                "Plantae",
                "Amphibia",
                "Mammalia"
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Device Available : ",device)


    val_dataset = Custom_val_dataset(dataset_path = 'inaturalist_12K/val')
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CNN()
    model.to(device)

    model.load_state_dict(torch.load(MODEL_PATH))
    acc = calculate_accuracy(model,val_dataloader)

    print('Accuracy of the model based on the test set of inputs is: %.2f %%' % (acc))



    # num_classes = len(class_map)
    # images_by_class = [[] for _ in range(num_classes)]
    # true_labels_by_class = [[] for _ in range(num_classes)]
    # predicted_labels_by_class = [[] for _ in range(num_classes)]

    # with torch.no_grad():
    #     for sample_images, sample_labels in val_dataloader:
    #         sample_images = sample_images.to(device)
    #         sample_labels = sample_labels.to(device)

    #         predictions = model(sample_images)
    #         _, predicted = torch.max(predictions, 1)

    #         for i in range(sample_images.size(0)):
    #             true_label = sample_labels[i].item()
    #             predicted_label = predicted[i].item()

    #             images_by_class[true_label].append(sample_images[i].cpu())
    #             true_labels_by_class[true_label].append(true_label)
    #             predicted_labels_by_class[true_label].append(predicted_label)

    # plt.figure(figsize=(15, 30))
    # for i in range(num_classes):
    #     class_name = class_map[i]

    #     num_images_to_display = min(3, len(images_by_class[i]))
    #     for j in range(num_images_to_display):
    #         plt.subplot(num_classes, 3, i * 3 + j + 1)
    #         img_np = (images_by_class[i][j].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    #         plt.imshow(img_np)
    #         plt.title(f"True: {class_map[true_labels_by_class[i][j]]}\nPred: {class_map[predicted_labels_by_class[i][j]]}")
    #         plt.axis('off')

    # plt.tight_layout()

    # wandb.log({"sample_predictions": plt})

    # plt.savefig('sample_predictions.png')

    # wandb.finish()