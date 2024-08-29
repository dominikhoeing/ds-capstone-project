import pandas as pd
import matplotlib.pyplot as plt

def evaluation_metric(model, saving = False):
    
    """
    Plots and (optionally) saves accuracy and loss of the model per epoch.

    Args:
        model: The trained model object
        saving = False: if True, it asks for user input of a filename. The accuracies and losses are saved as a csv file in the current directory.

    Returns:
        plots of Training and Validation Accuracy
        saves result as csv file (optional)
    """

    # create variable for different accuracy metrics vs. epochs
    accuracy = model.history['accuracy']
    val_accuracy = model.history['val_accuracy']
    loss = model.history['loss']
    val_loss = model.history['val_loss']

    # plot Accuracy
    fig1, ax1 = plt.subplots()
    ax1.plot(accuracy,label='train_accuracy')
    ax1.plot(val_accuracy,label='val_accuracy')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('accuracy')
    ax1.legend()
    fig1.show()

    # plot Loss
    fig2, ax2 = plt.subplots()
    ax2.plot(loss,label='train_loss')
    ax2.plot(val_loss,label='val_loss')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('loss')
    ax2.legend()
    fig2.show()

    # saving the metrics in a csv-file
    if saving == True:
        d = {'accuracy': accuracy, 'val_accuracy': val_accuracy, 'loss': loss, 'val_loss': val_loss}
        df = pd.DataFrame(d)
        name = input("Please enter the filename (without .csv): ")
        name = name + '.csv'
        df.to_csv(name)