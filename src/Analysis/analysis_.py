# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1x0kh_PP33qObRvrtvvWZ2ENUJAqvwVqB
"""

import numpy as np
import bagpy
from math import remainder, tau
import math
from scipy.signal import butter, lfilter
import scipy
import rosbag
from sympy import Eq, solve, sin, cos
from sympy.abc import z, o
from skimage.measure import EllipseModel
import pandas as pd
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import scipy.integrate as integrate

bag = rosbag.Bag("data_lab4.bag")

latitudes = []
longitudes = []
magnetic_x = []
magnetic_y = []
magnetic_z = []
secs_gps = []
secs_imu = []
points = []
distances = []
theta = 0.0633214796217355
bag_gps = list(bag.read_messages(topics="/gps"))
bag_imu = list(bag.read_messages(topics="/imu"))

for i in range(len(bag_imu)):
    if i < len(bag_gps):
        latitudes.append(bag_gps[i][1].Latitude)
        longitudes.append(bag_gps[i][1].Longitude)
        secs_gps.append(bag_gps[i][1].header.stamp.secs)
    magnetic_x.append(bag_imu[i][1].MagField.magnetic_field.x)
    magnetic_y.append(bag_imu[i][1].MagField.magnetic_field.y)
    magnetic_z.append(bag_imu[i][1].MagField.magnetic_field.z)
    secs_imu.append(bag_imu[i][1].header.stamp.secs + float(bag_imu[i][1].header.stamp.nsecs) / (10**9))
xyz = int(len(bag_gps) / 30)
mid_latitude = np.mean(latitudes[xyz: xyz*3])
mid_longitude = np.mean(longitudes[xyz: xyz*3])
print(mid_latitude, mid_longitude)

for i in range(xyz, xyz*3):
    distances.append(np.sqrt((latitudes[i] - mid_latitude) ** 2 + (longitudes[i] - mid_longitude) ** 2))
    points.append((latitudes[i], longitudes[i]))
mean_distance = np.mean(distances)
t = np.linspace(0, 2 * np.pi, 100)

circle_timeframe = int(len(bag_imu) / 30)
magnetic_x = np.array(magnetic_x)
magnetic_y = np.array(magnetic_y)
plt.scatter(magnetic_x[circle_timeframe: circle_timeframe*3], magnetic_y[circle_timeframe: circle_timeframe*3])
plt.axis("equal")
plt.title("Magnetometer data before calibration")
plt.xlabel("Magnetic field X (gauss)")
plt.ylabel("Magnetic field Y (gauss)")
plt.show()
mid_mag_x = np.mean(magnetic_x[circle_timeframe: circle_timeframe*3])
mid_mag_y = np.mean(magnetic_y[circle_timeframe: circle_timeframe*3])
print(f"Hard iron x = {mid_mag_x}; Hard iron y = {mid_mag_y}")

distances = np.sqrt(magnetic_x[circle_timeframe: circle_timeframe*3] ** 2 + magnetic_y[circle_timeframe: circle_timeframe*3] ** 2)
mean_radius = np.mean(distances)
plt.plot(mean_radius * np.cos(t), mean_radius * np.sin(t), color="r")
plt.scatter(magnetic_x[circle_timeframe: circle_timeframe*3] - mid_mag_x, magnetic_y[circle_timeframe: circle_timeframe*3] - mid_mag_y)
plt.axis("equal")
plt.title("Magnetometer data after hard iron calibration")
plt.xlabel("Magnetic field X (gauss)")
plt.ylabel("Magnetic field Y (gauss)")
plt.show()

def calculate_ab():
    sorted_distances = np.sort(distances)
    b = np.mean(sorted_distances[:1000])
    a = np.mean(sorted_distances[-1000:])
    return a, b
a_value, b_value = calculate_ab()
temp_magnetic = []
print(a_value, b_value)
x_transformed = []
y_transformed = []

for i in range(len(bag_imu)):
    x_, y_ = np.matmul([[(a_value + b_value) * np.cos(theta) / (2 * a_value), -1 * (a_value + b_value) * np.sin(theta) / (2 * a_value)],
                       [(a_value + b_value) * np.sin(theta) / (2 * b_value), (a_value + b_value) * np.cos(theta) / (2 * b_value)]],
                      [magnetic_x[i] - mid_mag_x, magnetic_y[i] - mid_mag_y])
    x_transformed.append(x_)
    y_transformed.append(y_)
distances = np.sqrt(np.array(x_transformed[circle_timeframe: circle_timeframe*3]) ** 2 + np.array(y_transformed[circle_timeframe: circle_timeframe*3]) ** 2)
mean_radius = np.mean(distances)
plt.axhline(0, color='k', linestyle='--', linewidth=1)  # Perpendicular line parallel to the y-axis
plt.axvline(0, color='k', linestyle='--', linewidth=1)  #
plt.plot(mean_radius * np.cos(t), mean_radius * np.sin(t), color="r")
plt.scatter(x_transformed[circle_timeframe: circle_timeframe*3], y_transformed[circle_timeframe: circle_timeframe*3])
plt.title("Magnetometer data after soft iron calibration")
plt.xlabel("Magnetic field X (gauss)")
plt.ylabel("Magnetic field Y (gauss)")
plt.axis("equal")
plt.show()

gyro_x = []
gyro_y = []
gyro_z = []
yaw = []
pitch = []
roll = []

for i in range(len(bag_imu)):
    gyro_x.append(bag_imu[i][1].IMU.angular_velocity.x)
    gyro_y.append(bag_imu[i][1].IMU.angular_velocity.y)
    gyro_z.append(bag_imu[i][1].IMU.angular_velocity.z)
    raw_values = bag_imu[i][1].raw_values.split(',')
    yaw.append(float(raw_values[1].split("'")[1]))
    pitch.append(float(raw_values[2].split("'")[1]))
    roll.append(float(raw_values[3].split("'")[1]))

calibrated_yaw = np.arctan2(x_transformed, y_transformed)
yaw_integrated = np.cumsum(gyro_z)
plt.plot(yaw_integrated / 40)
plt.plot(np.unwrap(np.array(yaw) * np.pi / 180))
plt.plot(np.unwrap(calibrated_yaw))
plt.legend(["Yaw (Gyro Integration)", "Raw Yaw", "Calibrated Yaw"])
plt.title("Yaw Integration from Gyro Data")
plt.xlabel("Time")
plt.ylabel("Yaw Angle (radians)")
plt.show()

accelerometer_x = []
gps_timestamps = []
timestamps = []
northing_utm = []
easting_utm = []

for i in range(len(bag_imu)):
    accelerometer_x.append(bag_imu[i][1].IMU.linear_acceleration.x)
    seconds = bag_imu[i][1].header.stamp.secs
    nanoseconds = bag_imu[i][1].header.stamp.nsecs
    timestamps.append(float(seconds) + float(nanoseconds) / 10**9)

timestamps = np.array(timestamps) - timestamps[0]

for i in range(len(bag_gps)):
    northing_utm.append(bag_gps[i][1].northing_utm)
    easting_utm.append(bag_gps[i][1].Utm_easting)
    gps_seconds = bag_gps[i][1].header.stamp.secs
    gps_timestamps.append(gps_seconds)

gps_timestamps = np.array(gps_timestamps) - gps_timestamps[0]
gps_velocities = [np.sqrt((northing_utm[i + 1] - northing_utm[i])**2 + (easting_utm[i + 1] - easting_utm[i])**2) /
                  (gps_timestamps[i + 1] - gps_timestamps[i]) for i in range(len(bag_gps) - 1)]

adjusted_accelerometer_x = np.array(accelerometer_x)
time_ranges = [0, 125, 190, 265, 370, 420, 500, 670, 740, 828]

for i in range(len(time_ranges) - 1):
    start_point = time_ranges[i]
    end_point = time_ranges[i + 1]

    net_zero_acceleration = accelerometer_x[int(start_point * len(timestamps) / 828):int(end_point * len(timestamps) / 828)]
    adjusted_accelerometer_x[int(start_point * len(timestamps) / 828):int(end_point * len(timestamps) / 828)] = net_zero_acceleration - np.mean(net_zero_acceleration)

plt.plot(timestamps, adjusted_accelerometer_x)
plt.plot(timestamps, accelerometer_x)
plt.legend(["Adjusted Accelerometer (x)", "Raw Accelerometer (x)"])
plt.title("Acceleration vs. Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s^2)")
plt.show()
integrated_velocity = np.cumsum(np.array(adjusted_accelerometer_x))
integrated_velocity[integrated_velocity < 0] = 0
plt.plot((np.array(timestamps[:-1]) - timestamps[0]), (integrated_velocity[:-1]) / 40)
plt.plot(np.array(gps_timestamps[:-1]) - gps_timestamps[0], np.array(gps_velocities))
plt.legend(["Adjusted Forward Velocity", "GPS Velocity"])
plt.title("Forward Velocity vs. Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Velocity (m/s)")
plt.show()

timestamps = [0, 1, 2, 3, 4, 5]  # Replace with your timestamps
integrated_velocity = [0, 1, 2, 3, 4, 5]  # Replace with your integrated velocity data
gps_velocity = [0, 1, 2, 3, 4, 5]  # Replace with your GPS velocity data

# Align the timestamps and integrated_velocity arrays
timestamps = timestamps[:len(integrated_velocity)]

# Calculate displacement using integration
inte_integrated_velocity = integrate.cumtrapz(np.array(integrated_velocity) / 1600)
inte_gps_velocity = integrate.cumtrapz(np.array(gps_velocity))

# Create a time array for plotting
time = np.array(timestamps[1:-1]) - timestamps[0]

# Plot the results
plt.plot(time, inte_integrated_velocity, label="Displacement from IMU data")
plt.plot(time, inte_gps_velocity, label="Displacement from GPS data")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Displacement")
plt.title("Displacement Comparison")
plt.grid()
plt.show()

accel_y = []

for i in range(len(bag_imu)) :
    accel_y.append(bag_imu[i][1].IMU.linear_acceleration.y)
plt.plot(timestamps, accel_y)
plt.plot(np.array(timestamps[:-1])-timestamps[0], ((integrated_velocity)/40)*gyro_z[:-1])
plt.legend(["Linear acceleration y", "observed linear acceleration"])
plt.show()
cutoff = 0.001
fs = 40
nyq = 0.5 * fs
normal_cutoff = cutoff / nyq
b, a = butter(1, normal_cutoff, btype='low', analog=False)
filtered_lin_y = lfilter(b, a, accel_y)
lpf = scipy.signal.filtfilt(*butter(3, 0.1, "lowpass",fs = 40, analog=False), accel_y)
plt.plot(timestamps, lpf)
plt.plot(np.array(timestamps[:-1])-timestamps[0], ((integrated_velocity)/40)*gyro_z[:-1])
plt.legend(["Linear acceleration y with lpf", "observed linear acceleration"])
plt.show()

v_n = integrate.cumtrapz(np.array(integrated_velocity*np.sin(np.array(yaw)[:-1]*np.pi/180)))
v_e = integrate.cumtrapz(np.array(integrated_velocity*np.cos(np.array(yaw)[:-1]*np.pi/180)))
plt.plot(v_n/1600, v_e/1600)
plt.plot(np.array(easting_utm)-easting_utm[0], np.array(northing_utm)-northing_utm[0])
plt.xlabel("Position x (meters)")
plt.ylabel("Position y (meters)")
plt.legend(["Estimated trajectory", "GPS path"])
plt.show()

degree = -45
theta1 = yaw[0]*np.pi/180 + degree*np.pi/180
theta2 = np.arctan2(northing_utm[1]-northing_utm[0], easting_utm[1]-easting_utm[0])
rotation_matrix = [[np.cos(theta1-theta2), -1*np.sin(theta1-theta2)], [np.sin(theta1-theta2), np.cos(theta1-theta2)]]
v_n_rotated = []
v_e_rotated = []
for i in range(len(bag_imu)-2) :
    x_, y_ = np.matmul(rotation_matrix, [v_e[i], v_n[i]])
    v_e_rotated.append(x_)
    v_n_rotated.append(y_)
plt.plot(np.array(v_n_rotated)/1600, np.array(v_e_rotated)/1600)
plt.plot(np.array(easting_utm)-easting_utm[0], np.array(northing_utm)-northing_utm[0])
plt.legend(["Estimated trajectory (after adjustment)", "GPS path"])
plt.xlabel("Position x (meters)")
plt.ylabel("Position y (meters)")
plt.show()

