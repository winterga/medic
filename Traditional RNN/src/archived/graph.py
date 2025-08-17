import matplotlib.pyplot as plt
import numpy as np

# Function to plot the accuracy graph with a broken y-axis
def plot_accuracy_graph(epochs, train_acc_class0, train_acc_class1, train_acc_class2,
                        val_acc_class0, val_acc_class1, val_acc_class2,
                        train_overall, val_overall):
    # Create two figures for train and validation accuracy
    fig_train, (ax_high_train, ax_low_train) = plt.subplots(2, 1, sharex=True, figsize=(10, 7), gridspec_kw={'height_ratios': [5, 1]})
    fig_val, (ax_high_val, ax_low_val) = plt.subplots(2, 1, sharex=True, figsize=(10, 7), gridspec_kw={'height_ratios': [5, 1]})

    # Plot Training accuracies on the higher plot (0%-100%)
    ax_high_train.plot(epochs, train_acc_class0, label='Train Class 0', marker='o', lw=2)
    ax_high_train.plot(epochs, train_acc_class1, label='Train Class 1', marker='o', lw=2)
    ax_high_train.plot(epochs, train_acc_class2, label='Train Class 2', marker='o', lw=2)
    ax_high_train.plot(epochs, train_overall, label='Train Overall', lw=2, linestyle='--')

    # Plot Validation accuracies (including Class 2) on the higher plot
    ax_high_val.plot(epochs, val_acc_class0, label='Validation (0) Within Collecting System Anatomy', lw=2)
    ax_high_val.plot(epochs, val_acc_class1, label='Validation (1) Before Scope Insertion/After Scope Removal', lw=2)
    ax_high_val.plot(epochs, val_acc_class2, label='Validation (2) Menu Screens', lw=2)
    ax_high_val.plot(epochs, val_overall, label='Validation Overall', lw=2, linestyle='--')

    # Duplicate the lower part for the 0%-5% range for context
    ax_low_train.plot(epochs, train_acc_class0, marker='o', lw=2, alpha=0.3)
    ax_low_train.plot(epochs, train_acc_class1, marker='o', lw=2, alpha=0.3)
    ax_low_train.plot(epochs, train_acc_class2, marker='o', lw=2, alpha=0.3)
    ax_low_train.plot(epochs, train_overall, lw=2, alpha=0.3, linestyle='--')

    ax_low_val.plot(epochs, val_acc_class0, lw=2, alpha=0.3)
    ax_low_val.plot(epochs, val_acc_class1, lw=2, alpha=0.3)
    ax_low_val.plot(epochs, val_acc_class2, lw=2, alpha=0.3)
    ax_low_val.plot(epochs, val_overall, lw=2, alpha=0.3, linestyle='--')

    # Set limits for both plots (0%-100% range)
    ax_high_train.set_ylim(60, 100)  # Main focus range (0-100%)
    ax_high_train.set_yticks(range(60, 101, 5))  # 5% increments on high range
    ax_low_train.set_ylim(0, 5)  # Only show 0% to 5% for context

    ax_high_val.set_ylim(60, 100)  # Main focus range (0-100%)
    ax_high_val.set_yticks(range(60, 101, 5))  # 5% increments on high range
    ax_low_val.set_ylim(0, 5)  # Only show 0% to 5% for context

    # Add broken axis effect using diagonal lines
    d = .015  # Size of the diagonal break lines
    kwargs = dict(transform=ax_low_train.transAxes, color='k', clip_on=False)
    ax_low_train.plot((-d, +d), (-d, +d), **kwargs)  # Bottom break line
    ax_low_train.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Bottom break line

    kwargs = dict(transform=ax_high_train.transAxes, color='k', clip_on=False)
    ax_high_train.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Top break line
    ax_high_train.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Top break line

    kwargs = dict(transform=ax_low_val.transAxes, color='k', clip_on=False)
    ax_low_val.plot((-d, +d), (-d, +d), **kwargs)  # Bottom break line
    ax_low_val.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Bottom break line

    kwargs = dict(transform=ax_high_val.transAxes, color='k', clip_on=False)
    ax_high_val.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Top break line
    ax_high_val.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Top break line

    # Labels, title, and legend
    ax_high_train.set_ylabel('Accuracy (%)')
    ax_low_train.set_ylabel('')
    ax_high_train.legend()
    ax_high_val.set_ylabel('Accuracy (%)')
    ax_low_val.set_ylabel('')
    ax_high_val.legend()

    plt.xlabel('Epochs')

    # Show plots
    plt.show()

# Function to parse the input file and extract accuracy values for each class
def parse_epoch_data(file_path):
    epochs = []
    train_acc_class0, train_acc_class1, train_acc_class2 = [], [], []
    val_acc_class0, val_acc_class1, val_acc_class2 = [], [], []  # Added val_acc_class2 back
    train_overall, val_overall = [], []  # Added overall accuracy lists

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if line.startswith("Epoch"):
            epoch = int(line.split()[1].split('/')[0])
            epochs.append(epoch)

            # Extract training accuracy values
            train_class0 = float(lines[i + 4].split(":")[1].strip().replace('%', ''))
            train_class1 = float(lines[i + 5].split(":")[1].strip().replace('%', ''))
            train_class2 = float(lines[i + 6].split(":")[1].strip().replace('%', ''))
            train_overall_acc = float(lines[i + 8].split(":")[1].strip().replace('%', ''))

            train_acc_class0.append(train_class0)
            train_acc_class1.append(train_class1)
            train_acc_class2.append(train_class2)
            train_overall.append(train_overall_acc)

            # Extract validation accuracy values (including Class 2)
            val_class0 = float(lines[i + 11].split(":")[1].strip().replace('%', ''))
            val_class1 = float(lines[i + 12].split(":")[1].strip().replace('%', ''))
            val_class2 = float(lines[i + 13].split(":")[1].strip().replace('%', ''))
            val_overall_acc = float(lines[i + 14].split(":")[1].strip().replace('%', ''))*100

            val_acc_class0.append(val_class0)
            val_acc_class1.append(val_class1)
            val_acc_class2.append(val_class2)
            val_overall.append(val_overall_acc)

    return epochs, train_acc_class0, train_acc_class1, train_acc_class2, val_acc_class0, val_acc_class1, val_acc_class2, train_overall, val_overall


# Example usage:
file_path = '../fe_correct_val_1_balanced.txt'  # Replace with your actual file path

# Parse the data from the file
epochs, train_acc_class0, train_acc_class1, train_acc_class2, val_acc_class0, val_acc_class1, val_acc_class2, train_overall, val_overall = parse_epoch_data(file_path)

# Plot the graph with the broken y-axis and overall accuracy included
plot_accuracy_graph(epochs, train_acc_class0, train_acc_class1, train_acc_class2,
                    val_acc_class0, val_acc_class1, val_acc_class2,
                    train_overall, val_overall)
