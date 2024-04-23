# emsemble.py
import ast
import logging
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from spoter.utils import train_epoch, evaluate, evaluate_checkpoints
from datasets.czech_slr_dataset import CzechSLRDataset
from datasets.data_processing import LandmarkDataset
from torch.utils.data import DataLoader

def get_top_checkpoint_predictions(top_checkpoint_name: str, eval_loader, device):
    ''' Get prediction tensor from a checkpointed model (intended to be the best model).
    :top_check_point_name: String, name of checkpoint file ('exp_name/checkpoint/checkpoint_id_v_i') to get predictions of.
    :return: Tensor of gloss predictions from the model.
    '''
    top_model = torch.load("out-checkpoints/" + top_checkpoint_name + ".pth") 
    top_model.train(False)
    predictions = []
    for i, data in enumerate(eval_loader):
        inputs, labels = data
        inputs = inputs.float()
        inputs = inputs.squeeze(0).to(device)
        outputs = top_model(inputs).expand(1, -1, -1) # Get output tensor, expand to 1x(#glosses)x1
        predictions.append(outputs)
        # print("outputs", outputs.shape)
    print("top checkpoint predictions", len(predictions), predictions[0].shape)

    return predictions

def get_metrics(vp_pred, mp_pred, eval_loader, device):
    ''' Calculate ensemble predicted classes, compute accuracy, generate confusion matrix.
    :param predictions: List tensor predictions.
    :return: Tuple of floats (pred_correct, pred_all, pred accuracy). '''
  
    pred_correct, pred_all = 0, 0
    stats = {i: [0, 0] for i in range(101)}
    predicted_classes = []
    true_labels = []
    for i, data in enumerate(eval_loader):
        inputs, labels = data
        inputs = inputs.float()
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)

        # Get the average prediction for the current gloss
        outputs = ensemble_predictions # predicted list for the current gloss
        # print("outputs, i", outputs, i)
        prediction = int(torch.argmax(torch.nn.functional.softmax(outputs, dim=1))) 
        # print("prediction", prediction)
        true_label = int(labels[0][0])
        predicted_classes.append(prediction)
        true_labels.append(true_label)
        print(f"Prediction: {prediction}, True label: {true_label}")
        
        if prediction == true_label:
            stats[true_label][0] += 1
            pred_correct += 1

        stats[true_label][1] += 1
        pred_all += 1

    stats = {key: value[0] / value[1] for key, value in stats.items() if value[1] != 0}
    print("ENSEMBLE Label accuracies statistics:")
    print(str(stats) + "\n")

    # Print a confusion matrix and classifcation report
    cm = create_confusion_matrix(predicted_classes, true_labels, num_classes=100)
    print("Confusion matrix:\n", cm)

    report = classification_report(true_labels, predicted_classes) # from scikit-learn
    print("Classification report:\n", report)

    return pred_correct, pred_all, (pred_correct / pred_all)

def create_confusion_matrix(predictions, true_labels, num_classes=100):
    """ Create a confusion matrix based on predictions and true labels.
    :param predictions: List of predicted labels (integers).
    :param true_labels: List of true labels.
    :param num_classes: Number of classes (glosses) in the dataset.
    :return: Confusion matrix as a 2D numpy array.
    """
    confusion_mat = np.zeros((num_classes, num_classes), dtype=int)
    for pred, true in zip(predictions, true_labels):
        confusion_mat[true, pred] += 1
    # Save confusion matrix as txt
    np.savetxt("confusion_matrix.txt", confusion_mat, fmt="%d")

    return confusion_mat

## PLAN:
# evaluate_checkpoints() for each model (each args.experiment_name)
   # will call evaluate() on the tested_model for each checkpoint, return best checkpointid

# evaluate_top_checkpoint() for each model, using best checkpoint returned
# but instead of getting eval_acc, get list of predictions of both top models

# then, use the predictions to calculate ensemble accuracy by averaging the predictions
# calculate ensemble accuracy and return it



if __name__ == '__main__':
    # Set device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("MPS is available")
        device = torch.device("mps")
        
    g = torch.Generator()
    g.manual_seed(379) # Same as default param of train.py

    # testing_set_path = 'WLASL100_test_25fps.csv' 
    # eval_set = CzechSLRDataset(testing_set_path) # CzechSLRDataset loads dataset of landmarks, has functions to normalize + augment
    testing_set_path = 'wlasl/test.csv' # Mediapipe
    val_set_path = 'wlasl/val.csv'
    eval_set = LandmarkDataset(val_set_path) # Mediapipe

    print("eval_set.data", len(eval_set.data))
    print("Dataset loaded")
    eval_loader = DataLoader(eval_set, shuffle=True, generator=g)
    
    # Get ensemble predictions and accuracy
    experiment_name = ['wlasl-100-mediapipe'] #'wlasl-100-visionapi'] # Folder name for the experiment's checkpoints

    top_vp_checkpoint = 'wlasl-100-visionapi/checkpoint_v_7' # already evaluated with evaluate_checkpoints
    top_vp_model = torch.load("out-checkpoints/" + top_vp_checkpoint + ".pth") 
    top_vp_model.train(False)
    # top_mp_checkpoint = 'wlasl-100-mediapipe/checkpoint_t_10'
    # top_mp_model = torch.load("out-checkpoints/" + top_mp_checkpoint + ".pth")
    # top_mp_model.train(False)

    vp_predictions = get_top_checkpoint_predictions(top_vp_checkpoint, eval_loader, device) # videos, 1, 100
    print("vp_predictions", vp_predictions.shape)
    # mp_predictions = get_top_checkpoint_predictions(top_mp_checkpoint, eval_loader, device)
    # print("mp_predictions", mp_predictions.shape)

    ens_pred_correct, ens_pred_all, ens_acc = evaluate(top_vp_model, eval_loader, device)
    print(f"Ensemble accuracy: {ens_acc}")
    print(f"Ensemble correct predictions: {ens_pred_correct}/{ens_pred_all}")




'''their loop through checkpoints'''
    # top_result = 0
    # top_result_name = ""
    # for i in range(11):
    #     for checkpoint_id in ["t", "v"]: # t -> train, v -> validation
    #         experiment_name = 'wlasl-100-mediapipe'
    #         tested_model = torch.load("out-checkpoints/" + experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i) + ".pth")
    #         tested_model.train(False)
    #         _, _, eval_acc = evaluate(tested_model, eval_loader, device, print_stats=True)

    #         if eval_acc > top_result:
    #             top_result = eval_acc
    #             top_result_name = experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i)

    #         print("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))

    # print("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")
    




    # todo
    # for experiment in experiment_name:
    #     print("Model", experiment)
    #     evaluate(experiment, eval_loader, device)
    #     model_predictions.append(get_top_checkpoint_predictions(top_checkpoint_name, eval_loader, device))

    
