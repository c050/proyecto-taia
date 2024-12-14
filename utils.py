import csv
import os


def save_metrics_to_csv(training_metrics, filename="training_metrics.csv"):
    if not os.path.exists("metrics"):
        os.makedirs("metrics")

    metrics_path = os.path.join("metrics", filename)
    """
    Store training metrics in a CSV file.

    Args:
        training_metrics (dict): Diccionario con las métricas de entrenamiento.
        filename (str): Nombre del archivo CSV (default: "metrics/training_metrics.csv").
    """

    epochs = range(1, len(training_metrics['loss_per_epoch']) + 1)

    with open(metrics_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Escribir encabezado
        header = ['Epoch', 'Train Loss', 'Test Loss', 'Training Time (s)']
        for i in range(6):
            header.append(f'MAE Component {i+1}')
        writer.writerow(header)

        # Escribir datos por época
        for epoch, train_loss, test_loss, train_time, mae_per_component in zip(
            epochs,
            training_metrics['loss_per_epoch'],
            training_metrics['test_loss_per_epoch'],
            training_metrics['training_time'],
            training_metrics['mae_per_component']
        ):
            row = [epoch, train_loss, test_loss, train_time]
            row.extend(mae_per_component)
            writer.writerow(row)
