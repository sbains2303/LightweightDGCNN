import pandas as pd
import numpy as np
import random

def verify_no_leakage(train, val, test):
    train_set = set(train)
    val_set = set(val)
    test_set = set(test)
    
    assert not train_set & val_set, f"Train/Val overlap: {train_set & val_set}"
    assert not train_set & test_set, f"Train/Test overlap: {train_set & test_set}"
    assert not val_set & test_set, f"Val/Test overlap: {val_set & test_set}"

def add_balanced_noise(route_df, typeData):
    """Add graduated noise based on purpose"""
    noisy_df = route_df.copy()
    sensor_cols = [col for col in route_df.columns if 'imu_' in col or 'ned_' in col]
    
    # Set noise levels
    if typeData == 'genuine':    
        scale = 0.05
        traj_scale = (0.95, 1.05)
    else:                      
        scale = 0.08
        traj_scale = (0.5, 1.0)
    
    # Apply noise
    for col in sensor_cols:
        if 'imu_' in col:
            noisy_df[col] += np.random.normal(0, scale * route_df[col].std(), len(route_df))
        elif 'ned_' in col:
            noisy_df[col] *= np.random.uniform(*traj_scale, len(route_df))
    
    return noisy_df

def generate_augmented_data(base_df, routes):
    augment = []
    for route in routes:
        genuine = base_df[base_df['route_number'] == route]
        spoofed = base_df[base_df['route_number'] == route]  
        
        if not genuine.empty:
            augment.append(add_balanced_noise(genuine, typeData='genuine'))
        if not spoofed.empty:
            augment.append(add_balanced_noise(spoofed, typeData='spoofed'))
    
    return pd.concat(augment) if augment else pd.DataFrame()


df = pd.read_csv("COMBINED_DATA")
route_categories = {
    'straight': [0, 1, 2, 3, 4],
    'circular': [5, 6, 7, 8, 9],
    'zigzag': [10, 11, 12, 13, 14],
    'random': [15, 16, 17, 18, 19],
    'rtl': [20, 21, 22, 23, 24],
    'payload': [25, 26, 27, 28, 29],
    'neighbourhood': [30, 31, 32, 33, 34],
    'transport': [35, 36, 37, 38, 39]
}

test_types = []
val_types = []
train_types = []

for category, routes in route_categories.items():
    selected = random.sample(routes, 2)  
    test_route = selected[0]
    val_route = selected[1]
    train_routes = [r for r in routes if r not in selected]

    test_types.append(test_route)
    val_types.append(val_route)
    train_types.extend(train_routes)

train_types += [r+40 for r in train_types]
val_types += [r+40 for r in val_types]
test_types += [r+40 for r in test_types]

print("\nRoute Type Assignments:")
print(f"Training routes: {len(train_types)//2} genuine + {len(train_types)//2} spoofed")
print(f"Validation routes: {len(val_types)//2} genuine + {len(val_types)//2} spoofed")
print(f"Test routes: {len(test_types)//2} genuine + {len(test_types)//2} spoofed")

train_augmented = generate_augmented_data(df, train_types, "training")
val_augmented = generate_augmented_data(df, val_types, "validation")
test_augmented = generate_augmented_data(df, test_types, "testing")

final_train = pd.concat([
    df[df['route_number'].isin(train_types)],
    train_augmented
])

final_val = pd.concat([
    df[df['route_number'].isin(val_types)],
    val_augmented
])

final_test = pd.concat([
    df[df['route_number'].isin(test_types)],
    test_augmented
])

verify_no_leakage(train_types, val_types, test_types)

final_train.to_csv("TRAIN", index=False)
final_val.to_csv("VAL", index=False)
final_test.to_csv("TEST", index=False)

print("\nFinal Dataset Sizes:")
print(f"Training: {len(final_train):,} samples")
print(f"Validation: {len(final_val):,} samples")
print(f"Testing: {len(final_test):,} samples")
