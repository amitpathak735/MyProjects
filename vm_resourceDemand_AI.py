# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pyVim import connect
from pyVmomi import vim
import ssl
from datetime import datetime, timedelta

# vCenter Connection Details (Replace with your actual details)
VCENTER_HOST = "your_vcenter_ip_or_hostname"
VCENTER_USER = "your_vcenter_username"
VCENTER_PASSWORD = "your_vcenter_password"
CLUSTER_NAME = "your_cluster_name"

# Data Collection Parameters
LOOKBACK_HOURS = 72  # Collect data for the last 72 hours
COLLECTION_INTERVAL_MINUTES = 5  # Collect data every 5 minutes

# Deep Learning Model Parameters
SEQUENCE_LENGTH = 24  # Look back at the previous 24 data points (adjust as needed)
PREDICTION_TIMESTEPS = 6 # Predict for the next 6 data points (e.g., 30 minutes if interval is 5 min)
TEST_SIZE = 0.2
EPOCHS = 50
BATCH_SIZE = 32

def collect_vm_performance_data(si, cluster_name, lookback_hours, interval_minutes):
    """Collects CPU and memory performance data for all VMs in a cluster."""
    content = si.RetrieveContent()
    cluster = None
    for c in content.rootFolder.childEntity:
        if hasattr(c, 'hostFolder'):
            for host in c.hostFolder.childEntity:
                if host.name == cluster_name:
                    cluster = host
                    break
            if cluster:
                break
    if not cluster:
        print(f"Error: Cluster '{cluster_name}' not found.")
        return {}

    vm_data = {}
    perf_manager = content.perfManager
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=lookback_hours)
    interval_seconds = interval_minutes * 60

    for vm in cluster.vm:
        print(f"Collecting data for VM: {vm.name}")
        perf_dict = {}
        query = perf_manager.QueryPerf(
            entity=vm,
            startTime=start_time,
            endTime=end_time,
            intervalId=interval_seconds
        )
        if query:
            for val in query:
                timestamp = val.sampleInfo[-1].timestamp.replace(tzinfo=None)
                cpu_usage = None
                mem_usage = None
                for metric in val.value:
                    if metric.id.counterId == 7:  # CPU Usage (%)
                        cpu_usage = metric.value[-1]
                    elif metric.id.counterId == 11: # Memory Usage (KB)
                        mem_usage = metric.value[-1] / 1024 # Convert to MB

                if cpu_usage is not None and mem_usage is not None:
                    if vm.name not in perf_dict:
                        perf_dict[vm.name] = {'Timestamp': [], 'CPU_Usage_Percent': [], 'Memory_Usage_MB': [], 'Configured_CPU_Cores': vm.config.hardware.numCPU, 'Configured_Memory_GB': vm.config.hardware.memoryMB / 1024}
                    perf_dict[vm.name]['Timestamp'].append(timestamp)
                    perf_dict[vm.name]['CPU_Usage_Percent'].append(cpu_usage)
                    perf_dict[vm.name]['Memory_Usage_MB'].append(mem_usage)

        if vm.name in perf_dict and perf_dict[vm.name]['Timestamp']:
            vm_data[vm.name] = pd.DataFrame(perf_dict[vm.name])
            vm_data[vm.name].sort_values(by='Timestamp', inplace=True)
        else:
            print(f"No performance data found for VM: {vm.name}")

    return vm_data

def preprocess_data(vm_df):
    """Preprocesses the performance data for a single VM."""
    if vm_df is None or vm_df.empty:
        return None, None, None, None

    vm_df['Hour'] = vm_df['Timestamp'].dt.hour
    vm_df['DayOfWeek'] = vm_df['Timestamp'].dt.dayofweek

    cpu_features = ['CPU_Usage_Percent', 'Configured_CPU_Cores', 'Hour', 'DayOfWeek']
    mem_features = ['Memory_Usage_MB', 'Configured_Memory_GB', 'Hour', 'DayOfWeek']

    cpu_scaler = MinMaxScaler()
    vm_df[cpu_features] = cpu_scaler.fit_transform(vm_df[cpu_features])

    mem_scaler = MinMaxScaler()
    vm_df[mem_features] = mem_scaler.fit_transform(vm_df[mem_features])

    return vm_df[cpu_features], vm_df['CPU_Usage_Percent'], vm_df[mem_features], vm_df['Memory_Usage_MB'], cpu_scaler, mem_scaler

def create_sequences(data, target, sequence_length, prediction_timesteps):
    """Creates sequences and corresponding future targets for time series prediction."""
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length - prediction_timesteps + 1):
        sequences.append(data[i : i + sequence_length])
        targets.append(target[i + sequence_length : i + sequence_length + prediction_timesteps])
    return np.array(sequences), np.array(targets)

def create_lstm_model(input_shape, output_dim=1):
    """Creates an LSTM model for time series prediction."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(prediction_timesteps) # Predict for multiple timesteps
    ])
    return model

if __name__ == "__main__":
    sslContext = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    sslContext.verify_mode = ssl.CERT_NONE

    try:
        si = connect.SmartConnectNoSSL(
            host=VCENTER_HOST,
            user=VCENTER_USER,
            pwd=VCENTER_PASSWORD,
            sslContext=sslContext
        )

        vm_performance_data = collect_vm_performance_data(si, CLUSTER_NAME, LOOKBACK_HOURS, COLLECTION_INTERVAL_MINUTES)

        vm_cpu_models = {}
        vm_mem_models = {}
        vm_cpu_scalers = {}
        vm_mem_scalers = {}
        vm_cpu_histories = {}
        vm_mem_histories = {}

        for vm_name, vm_df in vm_performance_data.items():
            print(f"\nProcessing data for VM: {vm_name}")
            cpu_features, cpu_target, mem_features, mem_target, cpu_scaler, mem_scaler = preprocess_data(vm_df.copy()) # Use a copy to avoid modifying the original DataFrame

            if cpu_features is not None:
                vm_cpu_scalers[vm_name] = cpu_scaler
                cpu_sequences, cpu_targets = create_sequences(cpu_features.values, cpu_target.values, SEQUENCE_LENGTH, PREDICTION_TIMESTEPS)
                X_train_cpu, X_test_cpu, y_train_cpu, y_test_cpu = train_test_split(cpu_sequences, cpu_targets, test_size=TEST_SIZE, shuffle=False)

                cpu_input_shape = (X_train_cpu.shape[1], X_train_cpu.shape[2])
                cpu_model = create_lstm_model(cpu_input_shape)
                cpu_model.compile(optimizer='adam', loss='mse')
                print(f"Training CPU model for {vm_name}...")
                cpu_history = cpu_model.fit(X_train_cpu, y_train_cpu, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test_cpu, y_test_cpu), verbose=0)
                vm_cpu_models[vm_name] = cpu_model
                vm_cpu_histories[vm_name] = cpu_history

            if mem_features is not None:
                vm_mem_scalers[vm_name] = mem_scaler
                mem_sequences, mem_targets = create_sequences(mem_features.values, mem_target.values, SEQUENCE_LENGTH, PREDICTION_TIMESTEPS)
                X_train_mem, X_test_mem, y_train_mem, y_test_mem = train_test_split(mem_sequences, mem_targets, test_size=TEST_SIZE, shuffle=False)

                mem_input_shape = (X_train_mem.shape[1], X_train_mem.shape[2])
                mem_model = create_lstm_model(mem_input_shape)
                mem_model.compile(optimizer='adam', loss='mse')
                print(f"Training Memory model for {vm_name}...")
                mem_history = mem_model.fit(X_train_mem, y_train_mem, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test_mem, y_test_mem), verbose=0)
                vm_mem_models[vm_name] = mem_model
                vm_mem_histories[vm_name] = mem_history

        # Prediction
        print("\nMaking Predictions:")
        for vm_name in vm_performance_data:
            if vm_name in vm_cpu_models and vm_name in vm_mem_models:
                last_cpu_sequence = vm_performance_data[vm_name][['CPU_Usage_Percent', 'Configured_CPU_Cores', 'Hour', 'DayOfWeek']].tail(SEQUENCE_LENGTH).values
                last_mem_sequence = vm_performance_data[vm_name][['Memory_Usage_MB', 'Configured_Memory_GB', 'Hour', 'DayOfWeek']].tail(SEQUENCE_LENGTH).values

                if len(last_cpu_sequence) == SEQUENCE_LENGTH and len(last_mem_sequence) == SEQUENCE_LENGTH:
                    scaled_cpu_sequence = vm_cpu_scalers[vm_name].transform(last_cpu_sequence)
                    scaled_mem_sequence = vm_mem_scalers[vm_name].transform(last_mem_sequence)

                    predicted_cpu_scaled = vm_cpu_models[vm_name].predict(np.expand_dims(scaled_cpu_sequence, axis=0))[0]
                    predicted_mem_scaled = vm_mem_models[vm_name].predict(np.expand_dims(scaled_mem_sequence, axis=0))[0]

                    # Inverse transform predictions
                    dummy_cpu_input = np.zeros((1, 4))
                    dummy_cpu_input[0, :1] = predicted_cpu_scaled[-1] # Use the last prediction for inverse transform
                    predicted_cpu_usage = vm_cpu_scalers[vm_name].inverse_transform(dummy_cpu_input)[0][0]

                    dummy_mem_input = np.zeros((1, 4))
                    dummy_mem_input[0, :1] = predicted_mem_scaled[-1] # Use the last prediction for inverse transform
                    predicted_mem_usage = vm_mem_scalers[vm_name].inverse_transform(dummy_mem_input)[0][0]

                    print(f"\n--- {vm_name} ---")
                    print(f"Predicted CPU Usage (next {PREDICTION_TIMESTEPS * COLLECTION_INTERVAL_MINUTES} mins): {predicted_cpu_usage:.2f}%")
                    print(f"Configured CPU Cores: {vm_performance_data[vm_name]['Configured_CPU_Cores'].iloc[0]}")
                    # Estimate CPU core demand (simple threshold-based logic)
                    if predicted_cpu_usage > 80:
                        estimated_needed_cores = np.ceil((predicted_cpu_usage / 100) * vm_performance_data[vm_name]['Configured_CPU_Cores'].iloc[0])
                        print(f"Estimated Potential CPU Cores Demand: {estimated_needed_cores:.0f}")
                    else:
                        print("Estimated Potential CPU Cores Demand: Adequate")

                    print(f"Predicted Memory Usage (next {PREDICTION_TIMESTEPS * COLLECTION_INTERVAL_MINUTES} mins): {predicted_mem_usage:.2f} MB")
                    print(f"Configured Memory: {vm_performance_data[vm_name]['Configured_Memory_GB'].iloc[0]:.2f} GB")
                    # Estimate memory demand (simple threshold-based logic)
                    if predicted_mem_usage > 0.8 * (vm_performance_data[vm_name]['Configured_Memory_GB'].iloc[0] * 1024):
                        print("Estimated Potential Memory Demand Increase")
                    else:
                        print("Estimated Potential Memory Demand: Adequate")
                else:
                    print(f"Not enough data to make predictions for {vm_name}")

    except vmodl.MethodFault as error:
        print(f"Caught vmodl error : {error.msg}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'si' in locals() and si:
            connect.Disconnect(si)