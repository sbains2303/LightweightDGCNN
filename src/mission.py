import asyncio
import json
import time
from mavsdk import System
from datetime import datetime, UTC
from mavsdk.mission import MissionItem, MissionPlan

# Function to convert IMU data to dictionary
def imu_to_dict(imu):
    return {
        "acceleration_frd": {
            "forward_m_s2": imu.acceleration_frd.forward_m_s2,
            "right_m_s2": imu.acceleration_frd.right_m_s2,
            "down_m_s2": imu.acceleration_frd.down_m_s2
        },
        "angular_velocity_frd": {
            "forward_rad_s": imu.angular_velocity_frd.forward_rad_s,
            "right_rad_s": imu.angular_velocity_frd.right_rad_s,
            "down_rad_s": imu.angular_velocity_frd.down_rad_s
        },
        "magnetic_field_frd": {
            "forward_gauss": imu.magnetic_field_frd.forward_gauss,
            "right_gauss": imu.magnetic_field_frd.right_gauss,
            "down_gauss": imu.magnetic_field_frd.down_gauss
        },
        "temperature_degc": imu.temperature_degc
    }

def velocity_ned_to_dict(velocity_ned):
    """ Convert VelocityNed object to a dictionary """
    return {
        "north_m_s": velocity_ned.north_m_s,
        "east_m_s": velocity_ned.east_m_s,
        "down_m_s": velocity_ned.down_m_s,
    }


def pressure_to_dict(pressure):
    return {
        "absolute_pressure_hpa": pressure.absolute_pressure_hpa,
        "temperature_deg": pressure.temperature_deg,
    }


# Function to load route data from a .plan file
def load_route_from_file(file_path):
    with open(file_path, "r") as file:
        route_data = json.load(file)
    return route_data["waypoints"]

# Function to collect IMU data
async def collect_imu_data(drone, data_list):
    print("Starting IMU data collection...")
    async for imu in drone.telemetry.imu():
        timestamp =   datetime.now(UTC).isoformat()
        data_list.append({'timestamp': timestamp, 'imu': imu_to_dict(imu)})
        await asyncio.sleep(1)

# Function to collect pressure data
async def collect_pressure_data(drone, data_list):
    print("Starting pressure data collection...")
    async for pressure in drone.telemetry.scaled_pressure():
        timestamp =  datetime.now(UTC).isoformat()
        data_list.append({'timestamp': timestamp, 'pressure': pressure_to_dict(pressure)})
        await asyncio.sleep(1)

# Function to collect actuator status data
async def collect_ned_data(drone, data_list):
    print("Starting NED data collection...")
    async for ned in drone.telemetry.velocity_ned():
        timestamp =  datetime.now(UTC).isoformat()
        data_list.append({'timestamp': timestamp, 'ned': velocity_ned_to_dict(ned)})
        await asyncio.sleep(1)

# Function to collect and save telemetry data
async def collect_telemetry_data(drone):
    print("Starting telemetry collection...")
    telemetry_data = []

    # Run all telemetry collection functions concurrently
    imu_task = asyncio.create_task(collect_imu_data(drone, telemetry_data))
    pressure_task = asyncio.create_task(collect_pressure_data(drone, telemetry_data))
    ned_task = asyncio.create_task(collect_ned_data(drone, telemetry_data))

    return imu_task, pressure_task, ned_task, telemetry_data

# Function to stop telemetry collection
async def stop_telemetry_collection(tasks):
    for task in tasks:
        task.cancel()

# Function to save telemetry data
def save_telemetry_data(telemetry_data):
    if telemetry_data:
        print(f"Collected {len(telemetry_data)} records.")
        with open('data/telemetry_data_79_S.json', 'w') as json_file:
            json.dump(telemetry_data, json_file, indent=4)
        print("Telemetry collection completed and saved to telemetry_data_79_S.json.")
    else:
        print("No telemetry data was collected.")

async def run():
    # Initialize drone connection
    drone = System()
    await drone.connect(system_address="udp://:PORT")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone!")
            break

    # Load waypoints
    waypoints = load_route_from_file("..routes/PLAN")
    print(f"Loaded {len(waypoints)} waypoints")

    # Create mission items from waypoints
    mission_items = []
    for wp in waypoints:
        mission_item = MissionItem(
            latitude_deg=wp["lat"],
            longitude_deg=wp["lon"],
            relative_altitude_m=wp["elevation"],
            speed_m_s=5,
            is_fly_through=True,
            gimbal_pitch_deg=0,
            gimbal_yaw_deg=0,
            camera_action=MissionItem.CameraAction.NONE,
            loiter_time_s=0,
            acceptance_radius_m=2,
            yaw_deg=float('nan'),  
            camera_photo_distance_m=0,
            camera_photo_interval_s=0,
            vehicle_action=MissionItem.VehicleAction.NONE
        )
        mission_items.append(mission_item)

    mission_plan = MissionPlan(mission_items)

    # Upload mission to drone
    print("-- Uploading mission")
    await drone.mission.upload_mission(mission_plan)

    # Arm the drone
    print("-- Arming")
    await drone.action.arm()

    # Start collecting telemetry data
    print("-- Starting telemetry collection")
    imu_task, pressure_task, ned_task, telemetry_data = await collect_telemetry_data(drone)

    # Start mission
    print("-- Starting mission")
    await drone.mission.start_mission()

    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > 300:  
            print("Time limit exceeded. Returning to launch.")
            await drone.action.return_to_launch()
            break
        await asyncio.sleep(1)  

    print("-- Stopping telemetry collection.")
    await stop_telemetry_collection([imu_task, pressure_task, ned_task])

    # Save telemetry data after mission is complete
    save_telemetry_data(telemetry_data)

    # Ensure the drone returns to launch after mission
    print("-- Returning to launch")
    await drone.action.return_to_launch()

if __name__ == "__main__":
    asyncio.run(run())
