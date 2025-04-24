import json
import pandas as pd
import numpy as np
import os

def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def process_sensor_data(json_data):
    combined_nodes = []
    i = 0

    while i < len(json_data):
        node = {}
        current_imu = None
        current_ned = None
        current_pressure = None

        entry_1 = json_data[i]
        timestamp = entry_1.get("timestamp", "unknown")
        if "imu" in entry_1:
            current_imu = entry_1["imu"]
        if "ned" in entry_1:
            current_ned = entry_1["ned"]
        if "pressure" in entry_1:
            current_pressure = entry_1["pressure"]

        if i + 1 < len(json_data):
            entry_2 = json_data[i + 1]
            if "imu" in entry_2:
                current_imu = entry_2["imu"]
            if "ned" in entry_2:
                current_ned = entry_2["ned"]
            if "pressure" in entry_2:
                current_pressure = entry_2["pressure"]

        if i + 2 < len(json_data):
            entry_3 = json_data[i + 2]
            if "imu" in entry_3:
                current_imu = entry_3["imu"]
            if "ned" in entry_3:
                current_ned = entry_3["ned"]
            if "pressure" in entry_3:
                current_pressure = entry_3["pressure"]

        if current_imu is None:
            print(f"Missing IMU data after timestamp {timestamp}")
        if current_ned is None:
            print(f"Missing NED data after timestamp {timestamp}")
        if current_pressure is None:
            print(f"Missing Pressure data after timestamp {timestamp}")

        node["imu_acceleration_frd_forward_m_s2"] = current_imu.get("acceleration_frd", {}).get("forward_m_s2", np.nan) if current_imu else np.nan
        node["imu_acceleration_frd_right_m_s2"] = current_imu.get("acceleration_frd", {}).get("right_m_s2", np.nan) if current_imu else np.nan
        node["imu_acceleration_frd_down_m_s2"] = current_imu.get("acceleration_frd", {}).get("down_m_s2", np.nan) if current_imu else np.nan

        node["imu_angular_velocity_frd_forward_rad_s"] = current_imu.get("angular_velocity_frd", {}).get("forward_rad_s", np.nan) if current_imu else np.nan
        node["imu_angular_velocity_frd_right_rad_s"] = current_imu.get("angular_velocity_frd", {}).get("right_rad_s", np.nan) if current_imu else np.nan
        node["imu_angular_velocity_frd_down_rad_s"] = current_imu.get("angular_velocity_frd", {}).get("down_rad_s", np.nan) if current_imu else np.nan

        node["imu_magnetic_field_frd_forward_gauss"] = current_imu.get("magnetic_field_frd", {}).get("forward_gauss", np.nan) if current_imu else np.nan
        node["imu_magnetic_field_frd_right_gauss"] = current_imu.get("magnetic_field_frd", {}).get("right_gauss", np.nan) if current_imu else np.nan
        node["imu_magnetic_field_frd_down_gauss"] = current_imu.get("magnetic_field_frd", {}).get("down_gauss", np.nan) if current_imu else np.nan

        node["imu_temperature_degc"] = current_imu.get("temperature_degc", np.nan) if current_imu else np.nan

        node["ned_north_m_s"] = current_ned.get("north_m_s", np.nan) if current_ned else np.nan
        node["ned_east_m_s"] = current_ned.get("east_m_s", np.nan) if current_ned else np.nan
        node["ned_down_m_s"] = current_ned.get("down_m_s", np.nan) if current_ned else np.nan

        node["pressure_absolute_pressure_hpa"] = current_pressure.get("absolute_pressure_hpa", np.nan) if current_pressure else np.nan
        node["pressure_temperature_deg"] = current_pressure.get("temperature_deg", np.nan) if current_pressure else np.nan

        combined_nodes.append(node)

        if current_imu and current_ned and current_pressure:
            i += 3
        elif current_imu and current_ned:
            i += 2
        else:
            i += 1

    return combined_nodes

def save_to_csv(df, file_name):
    df.to_csv(file_name, index=False)  

def convert_to_dataframe(combined_nodes):
    df = pd.DataFrame(combined_nodes)
    return df

data_dir = "DATA"
output_dir = "ORDERED"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(data_dir):
    if filename.endswith(".json"): 
        filepath = os.path.join(data_dir, filename)

        json_data = read_json(filepath)
        processed_data = process_sensor_data(json_data)
        df = convert_to_dataframe(processed_data)

        output_filename = f"{os.path.splitext(filename)[0]}.csv"  
        output_filepath = os.path.join(output_dir, output_filename)
        save_to_csv(df, output_filepath)
        print(f"Processed and saved: {output_filename}")

