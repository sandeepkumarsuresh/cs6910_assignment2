import torch
from preprocess import Custom_dataset
from torch.utils.data import DataLoader
from CNN import CNN

if __name__ == '__main__':

    BATCH_SIZE = 68
    MODEL_PATH = f'./model_{BATCH_SIZE}.pth'
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


    val_dataset = Custom_dataset(dataset_path = 'Train_Val_Dataset/val')
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # dataiter = iter(val_dataloader)
    # # images, labels = next(dataiter)

    # inputs, labels = dataiter[0].to(device), dataiter[1].to(device)

    net = CNN()
    net.to(device)

    net.load_state_dict(torch.load(MODEL_PATH))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in val_dataloader:
            # images, labels = data
            images, labels = data[0].to(device), data[1].to(device)

            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # correct_pred = {classname: 0 for classname in class_map}
    # total_pred = {classname: 0 for classname in class_map}

    # # again no gradients needed
    # with torch.no_grad():
    #     for data in val_dataloader:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predictions = torch.max(outputs, 1)
    #         # collect the correct predictions for each class
    #         for label, prediction in zip(labels, predictions):
    #             if label == prediction:
    #                 correct_pred[class_map[label]] += 1
    #             total_pred[class_map[label]] += 1


    # # print accuracy for each class
    # for classname, correct_count in correct_pred.items():
    #     accuracy = 100 * float(correct_count) / total_pred[classname]
    #     print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')