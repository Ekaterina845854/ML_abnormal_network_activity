import os
import json
import time
import logging
import traceback
import argparse
from datetime import datetime

import scapy.all as scapy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from model import Autoencoder
from capture import capture_traffic, process_pcap, extract_features

# Настройка логирования
logging.basicConfig(
    filename="anomalies.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Загрузка обученной модели
def load_model(model_path, input_dim):
    model = Autoencoder(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Предобработка данных
def preprocess_data(feature_matrix, scaler):
    features_scaled = scaler.transform(feature_matrix)
    return torch.tensor(features_scaled, dtype=torch.float32)

# Поиск аномалий
def detect_anomalies(model, data, threshold):
    model.eval()
    criterion = nn.MSELoss()
    anomalies = []
    reconstruction_errors = []

    with torch.no_grad():
        for i, x in enumerate(data):
            x = x.unsqueeze(0)
            reconstructed = model(x)
            loss = criterion(reconstructed, x).item()
            reconstruction_errors.append(loss)
            if loss > threshold:
                anomalies.append(i)

    return anomalies, reconstruction_errors

# Функция построения и сохранения графика аномалий
def save_anomaly_plot(reconstruction_errors, threshold, output_file="anomalies_plot.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(reconstruction_errors, label="Reconstruction Error", color="blue")
    plt.axhline(y=threshold, color="red", linestyle="--", label="Threshold")  

    plt.scatter(
        [i for i, error in enumerate(reconstruction_errors) if error > threshold], 
        [error for error in reconstruction_errors if error > threshold],
        color="red", label="Anomalies"
    )

    plt.xlabel("Sample Index")
    plt.ylabel("Reconstruction Error")
    plt.title("Anomaly Detection Results")
    plt.legend()
    plt.grid()
    plt.savefig(output_file)
    plt.close()
    print(f"📊 Anomaly plot saved to {output_file}")

# Функция записи и вывода аномалий
def save_anomalies(anomalies, reconstruction_errors, new_data, threshold, output_file="anomalies.json"):
    anomaly_records = [{"index": idx, "error": reconstruction_errors[idx], "data": new_data[idx].tolist()} for idx in anomalies]

    with open(output_file, "w") as f:
        json.dump(anomaly_records, f, indent=4)

    print(f"🔴 Anomalies saved to {output_file}")

    if anomalies:
        print("\n⚠️ Detected anomalies:")
        for idx in anomalies:
            print(f" - Index {idx}: Error {reconstruction_errors[idx]:.6f}")

    save_anomaly_plot(reconstruction_errors, threshold, output_file="anomalies_plot.png")

    return anomaly_records


# Основной цикл сбора и анализа данных
def continuous_capture(model, threshold, scaler, interval=60, output_dir="traffic_logs"):
    while True:
        try:
            pcap_file = capture_traffic(duration=interval, output_dir=output_dir)
            csv_file = process_pcap(pcap_file, output_dir=output_dir)

            df = pd.read_csv(csv_file)
            feature_matrix = extract_features(df)

            scaler.fit(feature_matrix)  
            new_data = preprocess_data(feature_matrix, scaler)

            anomalies, reconstruction_errors = detect_anomalies(model, new_data, threshold)
            anomaly_records = save_anomalies(anomalies, reconstruction_errors, new_data.numpy(), threshold) 

            for record in anomaly_records:
                log_message = f"Anomaly detected at index {record['index']}, Error: {record['error']:.6f}"
                logging.info(log_message)
                print(f"⚡ {log_message}")

            print(f"\n✅ Detected {len(anomalies)} anomalies out of {len(new_data)} samples.\n")

        except Exception as e:
            logging.error(f"❌ Error during capture: {e}")
            print(f"❌ Error: {e}")  
        time.sleep(1)


def process_dataset(dataset_path, model, threshold, scaler):
    try:
        df = pd.read_csv(dataset_path)
        feature_matrix = extract_features(df)

        scaler.fit(feature_matrix)  
        new_data = preprocess_data(feature_matrix, scaler)

        anomalies, reconstruction_errors = detect_anomalies(model, new_data, threshold)
        anomaly_records = save_anomalies(anomalies, reconstruction_errors, new_data.numpy(), threshold) 

        for record in anomaly_records:
            log_message = f"Anomaly detected at index {record['index']}, Error: {record['error']:.6f}"
            logging.info(log_message)
            print(f"⚡ {log_message}")

        print(f"\n✅ Detected {len(anomalies)} anomalies out of {len(new_data)} samples.\n")
    
    except Exception as e:
        logging.error(f"❌ Error during dataset processing: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Anomaly Detection Script")
    parser.add_argument("--live", action="store_true", help="Run in live capture mode")
    parser.add_argument("--dataset", type=str, help="Path to dataset (e.g. CIC-IDS2017) for batch processing")
    parser.add_argument("--threshold", type=float, default=1.45353392, help="Set the threshold for anomaly detection")

    args = parser.parse_args()

    model_path = "D://Infotecs//autoencoder_model.pth"
    input_dim = 6
    model = load_model(model_path, input_dim)

    scaler = StandardScaler()
    threshold = args.threshold  

    if args.live:
        print("🚀 Running in live capture mode...")
        continuous_capture(model, threshold, scaler, interval=60)
    elif args.dataset:
        print(f"🚀 Processing dataset: {args.dataset}...")
        process_dataset(args.dataset, model, threshold, scaler)
    else:
        print("❌ Please specify either --live or --dataset option.")
