import torch
from preprocess import Custom_val_dataset
from torch.utils.data import DataLoader
from CNN import CNN

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

    # dataiter = iter(val_dataloader)
    # # images, labels = next(dataiter)

    # inputs, labels = dataiter[0].to(device), dataiter[1].to(device)

    model = CNN()
    model.to(device)

    model.load_state_dict(torch.load(MODEL_PATH))

    # running_accuracy = 0
    # total = 0

    # model.eval()
    # with torch.no_grad():
    #     for data in val_dataloader:
    #         inputs, outputs = data
    #         inputs = inputs.to(device)
    #         outputs = outputs.to(device)

    #         predicted_outputs = model(inputs)

    #         _, predicted = torch.max(predicted_outputs, 1)

    #         total += outputs.size(0)
    #         running_accuracy += (predicted == outputs).sum().item()

    acc = calculate_accuracy(model,val_dataloader)

print('Accuracy of the model based on the test set of inputs is: %.2f %%' % (acc))

 

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