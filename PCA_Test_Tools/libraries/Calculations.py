# %%
import pandas as pd
import numpy as np
import os
from os import path
import pickle
from collections import defaultdict
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import scipy.stats as stats
import open3d as o3d
import logging
from scipy.stats import binom
import glob
import gzip
from pathlib import Path
import pathlib
import session_info        
import time
import bz2
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
import plotly.io as pio

from sklearn.linear_model import RANSACRegressor , LinearRegression
import open3d as o3d

import math

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig()
logger = logging.getLogger(' ')
logger.setLevel(logging.INFO)

# session_info.show()

def open3d_plot(dataframe):
    x = dataframe['x'].tolist()
    y = dataframe['y'].tolist()
    z = dataframe['z'].tolist()

    points = np.vstack((x, y, z)).transpose()
    # logger.info(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries_with_editing([pcd])

def calculate_z(x_value,coeff, intercept, y_value=0):
    a, b = coeff
    c = intercept - 1
    z_value = a * x_value + b * y_value + c
    return z_value

def angular_size(mean_range, target_width=1, target_height=1.4):
    # Height increased to add distance to ground
    horizontal_angular_size_radians = 2 * math.atan(target_width / (2 * mean_range))
    vertical_angular_size_radians = 2 * math.atan(target_height / (2 * mean_range))

    horizontal_angular_size_degrees = math.degrees(horizontal_angular_size_radians)
    vertical_angular_size_degrees = math.degrees(vertical_angular_size_radians)
    # this return is for the full target.
    return horizontal_angular_size_degrees, vertical_angular_size_degrees

# We use the df_valid_returns dataframe after velocity filtering here
def dbscan_default(sensor_1, eps = 0.2, samples = 10, plot=False):
    X = sensor_1.df[['x', 'y', 'z']].values

    # standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # DBSCAN modifiers!!
    eps = 0.2  # ~ distance between points to be considered a cluster, smaller = smaller distance....
    samples = 10  # ...10 works fine here
    dbscan = DBSCAN(eps=eps, min_samples=samples)

    dbscan.fit(X_scaled)
    labels = dbscan.labels_

    if plot:
        # Plotting the DBSCAN result
        # this plot can be done better but... it works for now
        traces = []
        unique_labels = set(labels)
        unique_labels.remove(-1) # to remove the noise points from plot, you might need them to debug so just comment out if needed
        for label in unique_labels:
            if label == -1:
                color = 'black'
            else:
                color = label / len(unique_labels)
                
            indices = (labels == label)
            trace = go.Scatter3d(
                x=X_scaled[indices, 0],
                y=X_scaled[indices, 1],
                z=X_scaled[indices, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=color,
                    opacity=0.8
                ),
                name=f'Cluster {label}'
            )
            traces.append(trace)

        layout = go.Layout(
            title='DBSCAN Clustering',
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z')
            )
        )

        fig = go.Figure(data=traces, layout=layout)
        pio.show(fig)

    estimated_cluster = np.bincount(labels[labels >= 0]).argmax()
    # logger.info(f' preselected value: {estimated_cluster}')
    
    # Label of the cluster that contains the required data... (cluster number from the last plot)
    # hover over plot to see the cluster number
    # noise is plotted as cluster -1
    label_to_extract = estimated_cluster

    indices_label = np.where(labels == label_to_extract)[0]
    data_label = X[indices_label]
    # dataset contains all the points for cluster selected
    dataset = pd.DataFrame({'x': data_label[:, 0], 'y': data_label[:, 1], 'z': data_label[:, 2]})
    # df_int2 contains all the data of the points x ,y and z plus all other columns from the original data.
    df_int2 = pd.merge(dataset, sensor_1.df, on=['x', 'y', 'z'], how='inner')
    return df_int2

def velocity_filter(sensor_1, selected_velocity, tolerance):    
    # Inputs
    # this is meters per seconds not miles per hour
    # tolerance = +/- selected_velocity value
    # 2 for SiLC & Voyant
    vel_min_boundary = selected_velocity - tolerance
    vel_max_boundary = selected_velocity + tolerance
    # print(f'{vel_min_boundary} {vel_max_boundary}')
    sensor_1.df = sensor_1.df[(sensor_1.df['velocity'] >= vel_min_boundary) & (sensor_1.df['velocity'] <= vel_max_boundary)]

def dbscan_extended(sensor_1, start_frame):
    # Test for SiLC, split df in 4 and run seperatly
    sensor_1.df = sensor_1.df[sensor_1.df['frame_idx'] >= start_frame]
    sensor_1.df = sensor_1.df[sensor_1.df['x'] <= 100]

    df_first_q  = pd.DataFrame()
    df_second_q = pd.DataFrame()
    df_third_q  = pd.DataFrame()
    df_fourth_q = pd.DataFrame()
    df_first_q  = sensor_1.df[:int(len(sensor_1.df)/4)]
    df_second_q = sensor_1.df[int(len(sensor_1.df)/4):int(len(sensor_1.df)/4)*2]
    df_third_q  = sensor_1.df[int(len(sensor_1.df)/4)*2:int(len(sensor_1.df)/4)*3]
    df_fourth_q = sensor_1.df[int(len(sensor_1.df)/4)*3:int(len(sensor_1.df)/4)*4]
    # We use the df_valid_returns dataframe after velocity filtering here
    X = df_first_q[['x', 'y', 'z']].values

    # standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # DBSCAN modifiers!!
    eps = 0.1  # ~ distance between points to be considered a cluster, smaller = smaller distance....
    samples = 10  # ...10 works fine here
    dbscan = DBSCAN(eps=eps, min_samples=samples)

    dbscan.fit(X_scaled)
    labels = dbscan.labels_

    label_to_extract = np.bincount(labels[labels >= 0]).argmax()

    indices_label = np.where(labels == label_to_extract)[0]
    data_label = X[indices_label]
    # dataset contains all the points for cluster selected
    dataset = pd.DataFrame({'x': data_label[:, 0], 'y': data_label[:, 1], 'z': data_label[:, 2]})
    # df_int2 contains all the data of the points x ,y and z plus all other columns from the original data.
    df_int2_first_q = pd.merge(dataset, sensor_1.df, on=['x', 'y', 'z'], how='inner')

    # We use the df_valid_returns dataframe after velocity filtering here
    X = df_second_q[['x', 'y', 'z']].values

    # standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # DBSCAN modifiers!!
    eps = 0.1   # ~ distance between points to be considered a cluster, smaller = smaller distance....
    samples = 10  # ...10 works fine here
    dbscan = DBSCAN(eps=eps, min_samples=samples)

    dbscan.fit(X_scaled)
    labels = dbscan.labels_

    label_to_extract = np.bincount(labels[labels >= 0]).argmax()

    indices_label = np.where(labels == label_to_extract)[0]
    data_label = X[indices_label]
    # dataset contains all the points for cluster selected
    dataset = pd.DataFrame({'x': data_label[:, 0], 'y': data_label[:, 1], 'z': data_label[:, 2]})
    # df_int2 contains all the data of the points x ,y and z plus all other columns from the original data.
    df_int2_second_q = pd.merge(dataset, sensor_1.df, on=['x', 'y', 'z'], how='inner')

    # We use the df_valid_returns dataframe after velocity filtering here
    X = df_third_q[['x', 'y', 'z']].values

    # standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # DBSCAN modifiers!!
    eps = 0.09  # ~ distance between points to be considered a cluster, smaller = smaller distance....
    samples = 30  # ...10 works fine here
    dbscan = DBSCAN(eps=eps, min_samples=samples)

    dbscan.fit(X_scaled)
    labels = dbscan.labels_

    label_to_extract = np.bincount(labels[labels >= 0]).argmax()

    indices_label = np.where(labels == label_to_extract)[0]
    data_label = X[indices_label]
    # dataset contains all the points for cluster selected
    dataset = pd.DataFrame({'x': data_label[:, 0], 'y': data_label[:, 1], 'z': data_label[:, 2]})
    # df_int2 contains all the data of the points x ,y and z plus all other columns from the original data.
    df_int2_third_q = pd.merge(dataset, sensor_1.df, on=['x', 'y', 'z'], how='inner')

    # We use the df_valid_returns dataframe after velocity filtering here
    X = df_fourth_q[['x', 'y', 'z']].values

    # standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # DBSCAN modifiers!!
    eps = 0.08   # ~ distance between points to be considered a cluster, smaller = smaller distance....
    samples = 30  # ...10 works fine here
    dbscan = DBSCAN(eps=eps, min_samples=samples)

    dbscan.fit(X_scaled)
    labels = dbscan.labels_

    label_to_extract = np.bincount(labels[labels >= 0]).argmax()

    indices_label = np.where(labels == label_to_extract)[0]
    data_label = X[indices_label]
    # dataset contains all the points for cluster selected
    dataset = pd.DataFrame({'x': data_label[:, 0], 'y': data_label[:, 1], 'z': data_label[:, 2]})
    # df_int2 contains all the data of the points x ,y and z plus all other columns from the original data.
    df_int2_fourth_q = pd.merge(dataset, sensor_1.df, on=['x', 'y', 'z'], how='inner')

    df_int2 = pd.concat([df_int2_first_q, df_int2_second_q, df_int2_third_q, df_int2_fourth_q], ignore_index=True)

    del df_first_q,  df_second_q, df_third_q, df_fourth_q, df_int2_first_q, df_int2_second_q, df_int2_third_q, df_int2_fourth_q
    return df_int2

""" 
We will plot all 100 frames of data (valid returns), to see the vehicle movement through the frames.
We will have 2 selection screens:
For the first one, please select the first couple of targets as they appear in the point cloud.
For the second one, please select the first target only.
"""
def dynamic_target_selection(sensor_1):
    # if sensor_1.sensor_type == "voyant":
    x = sensor_1.df['x'].tolist()
    y = sensor_1.df['y'].tolist()
    z = sensor_1.df['z'].tolist()

    points = np.vstack((x, y, z)).transpose()
    # logger.info(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries_with_editing([pcd])

    # Read points from cropped_1.ply and shapes data
    current_path, tail= os.path.split(os.getcwd())
    ply_file = os.path.join(current_path, "temp_files", "cropped_1.ply" )
    pcd = o3d.io.read_point_cloud(ply_file)
    selected_points = np.asarray(pcd.points).reshape((-1, 3))

    # Reads the data from "selected_points" and adds it to dataframe "sensor_1.df_selection"
    mylistx = list()
    mylisty = list()
    mylistz = list()
    df_sel = pd.DataFrame()

    for i in range(len(selected_points)):
        mylistx.append(selected_points[i][0])
        mylisty.append(selected_points[i][1])
        mylistz.append(selected_points[i][2])

    df_sel['x']  = mylistx
    df_sel['y']  = mylisty
    df_sel['z']  = mylistz

    x = df_sel['x'].tolist()
    y = df_sel['y'].tolist()
    z = df_sel['z'].tolist()

    points = np.vstack((x, y, z)).transpose()
    # logger.info(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries_with_editing([pcd])

    # Read points from cropped_1.ply and shapes data
    current_path, tail= os.path.split(os.getcwd())
    ply_file = os.path.join(current_path, "temp_files", "cropped_1.ply" )
    pcd = o3d.io.read_point_cloud(ply_file)
    selected_points = np.asarray(pcd.points).reshape((-1, 3))

    # Reads the data from "selected_points" and adds it to dataframe "sensor_1.df_selection"
    mylistx = list()
    mylisty = list()
    mylistz = list()
    df_sel = pd.DataFrame()

    for i in range(len(selected_points)):
        mylistx.append(selected_points[i][0])
        mylisty.append(selected_points[i][1])
        mylistz.append(selected_points[i][2])

    df_sel['x']  = mylistx
    df_sel['y']  = mylisty
    df_sel['z']  = mylistz

    avr_range = df_sel['y'].mean()
    sensor_1.df_selection = df_sel
    del df_sel

    # df_int will contain the information for the frame that contains the target fully
    df_int = pd.merge(sensor_1.df_selection, sensor_1.df, on=['x', 'y', 'z'], how='inner')
    # logger.info(f"first = {df_int}")

    min_x = df_int.min()['x']
    min_y = df_int.min()['y']
    min_z = df_int.min()['z']
    max_x = df_int.max()['x']
    max_y = df_int.max()['y']
    max_z = df_int.max()['z']

    # start frame tells us in which of the 100 frames we first see the full target.
    start_frame = df_int['frame_idx'].min()
    del df_int
    logger.info(f"Start frame = {start_frame}")
    return start_frame

def ransac(df_int2, plot=True):
    # df_test_ransac = df_int2[df_int2['x'] <= 40] #sensor.df_valid_returns
    df_test_ransac = df_int2

    points = df_test_ransac[['x', 'y', 'z']].to_numpy()
    X = points[:, :2]
    y = points[:, 2]

    ransac = RANSACRegressor(estimator=LinearRegression(fit_intercept=False)) # base_estimator=LinearRegression(fit_intercept=False)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    inlier_points = points[inlier_mask]

    inlier_cloud = o3d.geometry.PointCloud() 
    inlier_cloud.points = o3d.utility.Vector3dVector(inlier_points)


    coeff = ransac.estimator_.coef_
    intercept = ransac.estimator_.intercept_
    print(f"Equation: {coeff[0]}x + {coeff[1]}y + {intercept} = z")

    if plot:
        # this is only to plot both data and generated plane.
        def create_plane(a, b, c, x_range, y_range, resolution=10):
            x_vals = np.linspace(x_range[0], x_range[1], resolution)
            y_vals = np.linspace(y_range[0], y_range[1], resolution)
            x_grid, y_grid = np.meshgrid(x_vals, y_vals)
            z_grid = a * x_grid + b * y_grid + c
            return np.vstack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten())).T

        a, b = coeff
        c = intercept -1
        x_range = (df_test_ransac['x'].min(), df_test_ransac['x'].max())
        y_range = (df_test_ransac['y'].min(), df_test_ransac['y'].max())

        plane_points = create_plane(a, b, c, x_range, y_range)
        plane_pcd = o3d.geometry.PointCloud()
        plane_pcd.points = o3d.utility.Vector3dVector(plane_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([1, 0, 0])  # Red = point cloud
        inlier_cloud.paint_uniform_color([0, 1, 0])  # Green = inliers
        plane_pcd.paint_uniform_color([0, 0, 1])  # Blue = plane

        o3d.visualization.draw_geometries([pcd, inlier_cloud, plane_pcd])
    return coeff, intercept

# speed = "19mphR", "2mph", "22mph", "65mph"
# sensor = sensor class variable
def loc_filter(df_int2, start_frame, coeff, intercept, range_tolerance = 0.2, prefilter = 0.2, ):# sensor, speed
    df_int3 = df_int2
    df_test_frames = pd.DataFrame()
    df_test_frames_result = pd.DataFrame()
     # LUT above
    # range_tolerance = 0.2
    # prefilter = 0.2
    
    # if speed == "19mphR":
    #     if sensor.sensor_type == "voyant":      range_tolerance = 0.2 # "voyant"    19mphR
    #     elif sensor.sensor_type == "scantinel": range_tolerance = 0.2 # "scantinel" 19mphR
    #     elif sensor.sensor_type == "silc":      range_tolerance = 0.5 # "silc"      19mphR
    #     elif sensor.sensor_type == "HRL":       range_tolerance = 0.2 # "HRL"       19mphR
    #     prefilter = -2
    # elif speed == "2mph":
    #     if sensor.sensor_type == "voyant":      range_tolerance = 0.2 # "voyant"    2mph
    #     elif sensor.sensor_type == "scantinel": range_tolerance = 0.2 # "scantinel" 2mph
    #     elif sensor.sensor_type == "silc":      range_tolerance = 0.2 # "silc"      2mph
    #     elif sensor.sensor_type == "HRL":       range_tolerance = 0.2 # "HRL"       2mph
    #     prefilter = -.2
    # elif speed == "22mph":
    #     if sensor.sensor_type == "voyant":      range_tolerance = 0.2 # "voyant"    22mph
    #     elif sensor.sensor_type == "scantinel": range_tolerance = 0.2 # "scantinel" 22mph
    #     elif sensor.sensor_type == "silc":      range_tolerance = 0.5 # "silc"      22mph
    #     elif sensor.sensor_type == "HRL":       range_tolerance = 0.2 # "HRL"       22mph
    #     prefilter = -2
    # elif speed == "65mph":
    #     if sensor.sensor_type == "voyant":      range_tolerance = 0.2 # "voyant"    65mph
    #     elif sensor.sensor_type == "scantinel": range_tolerance = 0.2 # "scantinel" 65mph
    #     elif sensor.sensor_type == "silc":      range_tolerance = 1.5 # "silc"      65mph
    #     elif sensor.sensor_type == "HRL":       range_tolerance = 0.2 # "HRL"       65mph
    #     prefilter = -2



    #  TODO: add if
    # normal loop
    calc = range(df_int3['frame_idx'].max())
    iterate_value = abs(df_int3['frame_idx'].min() - start_frame)
    # reverse loop
    # iterate_value = df_int3['frame_idx'].min()
    # calc = range(start_frame - iterate_value)

    # Loop starts here!!!! increase iterate_value until max
    for i in calc:
        # iterate_value = 78 # iterate_value + 1
        # if iterate_value == df_int3['frame_idx'].max(): break
        # print(iterate_value)
        # get mean of points in horizontal

        # normal loop TODO: add if
        df_test_frames = df_int3[df_int3['frame_idx'] == df_int3['frame_idx'].min() + iterate_value]
        # reverse loop
        # df_test_frames = df_int3[df_int3['frame_idx'] == iterate_value]

        # print(df_int3['frame_idx'].min() + iterate_value)
        # mean_value_in_x = df_test_frames['x'].min()
        df_test_frames = df_test_frames[ (df_test_frames['x'] >= (df_test_frames['x'].median() - prefilter))] # -2

        median_value_in_horizontal = df_test_frames['azimuth_angle'].median()
        mean_value_in_range = df_test_frames['x'].mean()
        min_value_in_range = df_test_frames['x'].min()
        # print(f'median_value_in_horizontal = {median_value_in_horizontal}')
        # print(f'mean_value_in_range = {mean_value_in_range}')
        # print(f'min_value_in_range = {min_value_in_range}')

        horizontal_angular_size_degrees, vertical_angular_size_degrees = angular_size(mean_value_in_range, target_height=1.5)

        ground_z_start = calculate_z(mean_value_in_range, coeff=coeff , intercept= intercept)
        elevation_angle_rad = math.atan2(ground_z_start, mean_value_in_range)
        ground_point_deg = math.degrees(elevation_angle_rad)

        # Horizontal filter! # works for Voyant and SiLC
        df_test_frames = df_test_frames[(df_test_frames['azimuth_angle'] >= (median_value_in_horizontal - (horizontal_angular_size_degrees/2))) 
                                        & (df_test_frames['azimuth_angle'] <= (median_value_in_horizontal + (horizontal_angular_size_degrees/2)))]
        # help_x = df_test_frames['elevation_angle']
        # print(f' elevation_angle {help_x}') 
        # help_val_frame = df_test_frames['frame_idx'].mean()
        # help_val_start = df_test_frames['elevation_angle'].min()
        # help_val_end = df_test_frames['elevation_angle'].max()
        
        # Vertical filter! based on ground plane # works for Voyant and SiLC
        # print(f'frame = {help_val_frame} | target = {help_val_start:2f} -> {help_val_end:2f} | ground_point_deg =  {ground_point_deg:2f} < Target < {ground_point_deg + vertical_angular_size_degrees:2f}')
        df_test_frames = df_test_frames[(df_test_frames['elevation_angle'] >= (ground_point_deg)) 
                                        & (df_test_frames['elevation_angle'] <= (ground_point_deg + vertical_angular_size_degrees))]
        
        # # range filter! TODO make the filter depend on max min of range, not a static 0.5 value
        df_test_frames = df_test_frames[(df_test_frames['x'] <= (min_value_in_range + range_tolerance))]

        # print((min_value_in_range + range_tolerance))
        df_test_frames_result = pd.concat([df_test_frames_result, df_test_frames])
        
        if iterate_value == df_int3['frame_idx'].max(): break
        iterate_value = iterate_value + 1
        
    return df_test_frames_result

# %%
# seperates x, y and z and plots with open3D
def target_selection(sensor):

    x = sensor.df_single_frame['x'].tolist()
    y = sensor.df_single_frame['y'].tolist()
    z = sensor.df_single_frame['z'].tolist()
    # if sensor.sensor_type == "voyant":
    #     x = sensor.df_single_frame['x'].tolist()
    #     y = sensor.df_single_frame['y'].tolist()
    #     z = sensor.df_single_frame['z'].tolist()
        
    # elif sensor.sensor_type == "scantinel":
    #     x = sensor.df_single_frame['x'].tolist()
    #     y = sensor.df_single_frame['y'].tolist()
    #     z = sensor.df_single_frame['z'].tolist()
    #     # pass

    # elif sensor.sensor_type == "silc": # this is new
    #     x = sensor.df_single_frame['x'].tolist()
    #     y = sensor.df_single_frame['y'].tolist()
    #     z = sensor.df_single_frame['z'].tolist()

    # elif sensor.sensor_type == "HRL":
    #     x = sensor.df_single_frame['x'].tolist()
    #     y = sensor.df_single_frame['y'].tolist()
    #     z = sensor.df_single_frame['z'].tolist()
    
    # elif sensor.sensor_type == "SRL":
    #     x = sensor.df_single_frame['x'].tolist()
    #     y = sensor.df_single_frame['y'].tolist()
    #     z = sensor.df_single_frame['z'].tolist()
     
    # elif sensor.sensor_type == "Hessai":
    #     x = sensor.df_single_frame['x'].tolist()
    #     y = sensor.df_single_frame['y'].tolist()
    #     z = sensor.df_single_frame['z'].tolist()

    points = np.vstack((x, y, z)).transpose()
    # logger.info(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries_with_editing([pcd])

    # Read points from cropped_1.ply and shapes data
    #ply_file = os.path.join(f"{os.getcwd()}", "temp","cropped_1.ply")
    ply_file = os.path.join(f"{sensor.directory}","cropped_1.ply")
    pcd = o3d.io.read_point_cloud(ply_file)
    selected_points = np.asarray(pcd.points).reshape((-1, 3))
    logger.info('PLY file read from: %s',ply_file)

    # Reads the data from "selected_points" and adds it to dataframe "sensor.df_selection"
    mylistx = list()
    mylisty = list()
    mylistz = list()
    df_sel = pd.DataFrame()
   
    for i in range(len(selected_points)):
        mylistx.append(selected_points[i][0])
        mylisty.append(selected_points[i][1])
        mylistz.append(selected_points[i][2])

    df_sel['x']  = mylistx
    df_sel['y']  = mylisty
    df_sel['z']  = mylistz
    
    # fast way 
    # sensor.df_selection = pd.DataFrame(selected_points, columns=["x","y","z"])
    if sensor.sensor_type == "voyant":
        avr_range = df_sel['x'].mean()
        px.scatter(df_sel, x='y', y='z', color='x', width=800, height=800, title=f"{sensor.data_name}").write_html(f"{sensor.directory}{sensor.data_name}_{avr_range:.3f}m_graph.html")
        
    elif sensor.sensor_type == "scantinel":
        avr_range = df_sel['x'].mean()
        # px.scatter(df_sel, x='x', y='z', color='y', width=800, height=800, title=f"{data_name}").write_html(f"{sensor.directory}{data_name}_{avr_range:.3f}m_graph3.html")
        pass

    elif sensor.sensor_type == "silc":  # this is new
        avr_range = df_sel['x'].mean()
        px.scatter(df_sel, x='x', y='z', color='y', width=800, height=800, title=f"{sensor.data_name}").write_html(f"{sensor.directory}{sensor.data_name}_{avr_range:.3f}m_graph3.html")

    elif sensor.sensor_type == "HRL":
        avr_range = df_sel['x'].mean()
        px.scatter(df_sel, x='y', y='z', color='y', width=800, height=800, title=f"{sensor.data_name}").write_html(f"{sensor.directory}{sensor.data_name}_{avr_range:.3f}m_graph3.html")

    elif sensor.sensor_type == "SRL":
        avr_range = df_sel['x'].mean()
        px.scatter(df_sel, x='y', y='z', color='y', width=800, height=800, title=f"{sensor.data_name}").write_html(f"{sensor.directory}{sensor.data_name}_{avr_range:.3f}m_graph3.html")
    
    elif sensor.sensor_type == "Hessai":
        avr_range = df_sel['x'].mean()
        px.scatter(df_sel, x='y', y='z', color='y', width=800, height=800, title=f"{sensor.data_name}").write_html(f"{sensor.directory}{sensor.data_name}_{avr_range:.3f}m_graph3.html")
    
    # Outputs -------
    sensor.df_selection = df_sel
    #logger.info(sensor.df_selection)
    # -------
    
    if df_sel['x'].mean() > 0 and df_sel['y'].mean() > 0 and df_sel['z'].mean() > 0:
     logger.info(f' Target Selection Complete!')
    else: 
     logger.warning(f' Target Selection Empty') 
    del df_sel, avr_range, mylistx, mylisty, mylistz, points, pcd, selected_points, ply_file
    # return sensor.df_selection, avr_range

# %%
# For PD
def data_parting(sensor):
    df_all_frames_target = sensor.df

    if sensor.sensor_type == "voyant":
        df_filtered_first_frames = pd.DataFrame()
        keys = list(sensor.sensor.df_selection.columns.values)
        i1 = sensor.df_single_frame .set_index(keys).index
        i2 = sensor.df_selection.set_index(keys).index
        df_filtered_first_frames = sensor.df_single_frame[i1.isin(i2)]
        df_filtered_first_frames = df_filtered_first_frames[['point_idx']]

        df_all_frames_target = sensor.df_valid_returns[sensor.df_valid_returns['point_idx'].isin(df_filtered_first_frames['point_idx'])]
        avr_range = sensor.df_all_frames_target['x'].mean()
        px.scatter(df_all_frames_target, x='y', y='z', color='point_idx', width=800, height=800, title=f"{sensor.data_name}").write_html(f"{sensor.directory}{sensor.data_name}_{avr_range:.3f}m_plot.html")

        
    elif sensor.sensor_type == "scantinel":
        pass

    elif sensor.sensor_type == "silc": # this is new
    
        df_int = pd.merge(sensor.df_selection, sensor.df_valid_returns, on=['x', 'y', 'z'], how='inner')
        #logger.info(f"first = {df_int}")

        min_col = df_int.min()['col']
        min_row = df_int.min()['row']
        max_col = df_int.max()['col']
        max_row = df_int.max()['row']
        df_all_frames_target = pd.DataFrame()
        df_all_frames_target = sensor.df_valid_returns[  (sensor.df_valid_returns['col'] <= max_col) 
                                     & (sensor.df_valid_returns['col'] >= min_col) 
                                     & (sensor.df_valid_returns['row'] <= max_row) 
                                     & (sensor.df_valid_returns['row'] >= min_row)]
        avr_range = sensor.df_all_frames_target['y'].mean()
        px.scatter(df_all_frames_target, x='y', y='z', color='z', width=800, height=800, title=f"{sensor.data_name}").write_html(f"{sensor.directory}{sensor.data_name}_{avr_range:.3f}m_plot.html")


    elif sensor.sensor_type == "HRL":
       df_int = pd.merge(sensor.df_selection, sensor.df_valid_returns, on=['x', 'y', 'z'], how='inner')
       #logger.info(f"first = {df_int}") 
       min_col = df_int.min()['Laser Shot Cols']
       min_row = df_int.min()['Laser Shot Row']
       max_col = df_int.max()['Laser Shot Cols']
       max_row = df_int.max()['Laser Shot Row']
       df_all_frames_target = sensor.df_valid_returns[  (sensor.df_valid_returns['Laser Shot Cols'] <= max_col) 
                                     & (sensor.df_valid_returns['Laser Shot Cols'] >= min_col) 
                                     & (sensor.df_valid_returns['Laser Shot Row'] <= max_row) 
                                     & (sensor.df_valid_returns['Laser Shot Row'] >= min_row)]
       #logger.info(df_all_frames_target)
       #logger.info(list(df_all_frames_target.columns))
       avr_range = df_all_frames_target['x'].mean()
       px.scatter(df_all_frames_target, x='x', y='y', color='z', width=800, height=800, title=f"{sensor.data_name}").write_html(f"{sensor.directory}{sensor.data_name}_{avr_range:.3f}m_plot.html")

    elif sensor.sensor_type == "SRL":
       if sensor.debug:
        df_int = pd.merge(sensor.df_selection, sensor.df_valid_returns, on=['x', 'y', 'z'], how='inner')
       else: 
        df_int = pd.merge(sensor.df_selection, sensor.df, on=['x', 'y', 'z'], how='inner')
       
       min_col = df_int.min()['col']
       min_row = df_int.min()['row']
       max_col = df_int.max()['col']
       max_row = df_int.max()['row']

       if sensor.debug:
        df_all_frames_target = sensor.df_valid_returns[  (sensor.df_valid_returns['col'] <= max_col) 
                                     & (sensor.df_valid_returns['col'] >= min_col) 
                                     & (sensor.df_valid_returns['row'] <= max_row) 
                                     & (sensor.df_valid_returns['row'] >= min_row)]
        
       else:
        df_all_frames_target = sensor.df[  (sensor.df['col'] <= max_col) 
                                     & (sensor.df['col'] >= min_col) 
                                     & (sensor.df['row'] <= max_row) 
                                     & (sensor.df['row'] >= min_row)]
        
       df_all_frames_target=df_all_frames_target[df_all_frames_target['return_idx']==0].copy()
       df_slot=df_all_frames_target
       logger.info(df_slot) 
       ### Calculating outliers with IQR 1.5 Rule
       Q1=df_all_frames_target['range'].quantile(q=0.25)
       Q3=df_all_frames_target['range'].quantile(q=0.75)
       IQR=Q3-Q1
       High_Outlier= Q3+1.5*IQR
       Lower_Outlier= Q1-1.5*IQR
       df_all_frames_target_valid=df_all_frames_target[(df_all_frames_target['range']<=High_Outlier) & (df_all_frames_target['range']>=Lower_Outlier)]
       df_all_frames_target=df_all_frames_target_valid
       ##
       avr_range = df_all_frames_target['range'].mean()
       logger.info(avr_range)
       px.scatter(df_all_frames_target, x='x', y='y', color='z', width=800, height=800, title=f"{sensor.data_name}").write_html(f"{sensor.directory}{sensor.data_name}_{avr_range:.3f}m_plot.html")

    elif sensor.sensor_type == "Hessai":
       df_int = pd.merge(sensor.df_selection, sensor.df_valid_returns, on=['x', 'y', 'z'], how='inner')
       #logger.info(f"first = {df_int}") 
       min_col = df_int.min()['col']
       min_row = df_int.min()['row']
       max_col = df_int.max()['col']
       max_row = df_int.max()['row']
       df_all_frames_target = sensor.df_valid_returns[  (sensor.df_valid_returns['col'] <= max_col) 
                                     & (sensor.df_valid_returns['col'] >= min_col) 
                                     & (sensor.df_valid_returns['row'] <= max_row) 
                                     & (sensor.df_valid_returns['row'] >= min_row)]
       #logger.info(df_all_frames_target)
       #logger.info(list(df_all_frames_target.columns))
       avr_range = df_all_frames_target['x'].mean()
       px.scatter(df_all_frames_target, x='x', y='y', color='z', width=800, height=800, title=f"{sensor.data_name}").write_html(f"{sensor.directory}{sensor.data_name}_{avr_range:.3f}m_plot.html")

    # Outputs -------
    sensor.df = df_all_frames_target
    if sensor.debug:
        sensor.df_all_frames_target = df_all_frames_target # sensor.df_all_frames_target        
    #logger.info(sensor.df_all_frames_target)
    # -------
    logger.info(f' Data parting complete!')
    # return df_all_frames_target
    del min_col, min_row, max_col, max_row, df_all_frames_target

def data_grouping(sensor):
    df_return_counts = pd.DataFrame()
    df_frame_counts = pd.DataFrame()
    if sensor.sensor_type == "voyant":
        df_valid_returns_trim = sensor.df_all_frames_target[['point_idx', 'snr_linear', 'x' , 'y' , 'z']]
        df_return_counts = df_valid_returns_trim.groupby(['point_idx']).size().reset_index().rename(columns={0:'count'})
        df_return_counts_snr = df_valid_returns_trim.groupby(['point_idx'])['snr_linear'].mean().reset_index()
        df_return_counts["mean_snr"] = df_return_counts_snr['snr_linear']
        df_return_counts['count'] = df_return_counts['count']/3 # fix when only using 100
        df_return_counts["corrected_snr"] = (df_return_counts['mean_snr'] * df_return_counts['count'])/100

    elif sensor.sensor_type == "scantinel":
        df_valid_returns_trim = sensor.df_all_frames_target[['col', 'row', 'x' , 'y' , 'z']]
        df_return_counts = df_valid_returns_trim.groupby(['col', 'row']).size().reset_index().rename(columns={0:'count'})
        df_return_counts['count'] = df_return_counts['count']/100

        # Calculate standard deviation for each point (x, y, z) for each group
        df_std = df_valid_returns_trim.groupby(['col', 'row']).agg({'x': 'std', 'y': 'std', 'z': 'std'}).reset_index()

        # Merge the standard deviation dataframe with df_return_counts
        df_return_counts = pd.merge(df_return_counts, df_std, on=['col', 'row'])

        df_return_counts.fillna(0, inplace=True)

    elif sensor.sensor_type == "silc": # this is new
        df_valid_returns_trim = sensor.df_all_frames_target[['col', 'row', 'x' , 'y' , 'z', 'frame_idx']]
        # df_return_counts_trim = df_valid_returns_trim.groupby(['Laser Shot Row', 'Laser Shot Cols']).size().reset_index(name='count')
        # df_return_counts_trim['count'] = df_return_counts_trim['count'] / 100
        df_return_counts = df_valid_returns_trim.groupby(['col', 'row']).size().reset_index().rename(columns={0:'count'})
        df_frame_counts = df_valid_returns_trim.groupby(['frame_idx']).size().mean()
        df_return_counts['count'] = df_return_counts['count']/100  
        

    elif sensor.sensor_type == "HRL":
        df_valid_returns_trim = sensor.df_all_frames_target[['Laser Shot Row', 'Laser Shot Cols', 'x' , 'y' , 'z', 'frame_idx']]
        # df_return_counts_trim = df_valid_returns_trim.groupby(['Laser Shot Row', 'Laser Shot Cols']).size().reset_index(name='count')
        # df_return_counts_trim['count'] = df_return_counts_trim['count'] / 100
        df_frame_counts = df_valid_returns_trim.groupby(['frame_idx']).size().mean()
        df_return_counts = df_valid_returns_trim.groupby(['Laser Shot Row', 'Laser Shot Cols']).size().reset_index().rename(columns={0:'count'})
        df_return_counts['count'] = df_return_counts['count']/100

        
        # Merge the count information back into the original dataframe
        # df_valid_returns_trim = pd.merge(df_valid_returns_trim, df_return_counts_trim, on=['Laser Shot Row', 'Laser Shot Cols'], how='left')

    elif sensor.sensor_type == "SRL":
        if sensor.debug:
         df_valid_returns_trim = sensor.df_all_frames_target[['row', 'col', 'x' , 'y' , 'z', 'frame_idx']]
        else:
         df_valid_returns_trim = sensor.df[['row', 'col', 'x' , 'y' , 'z', 'frame_idx']]

        # df_return_counts_trim = df_valid_returns_trim.groupby(['Laser Shot Row', 'Laser Shot Cols']).size().reset_index(name='count')
        # df_return_counts_trim['count'] = df_return_counts_trim['count'] / 100
        df_frame_counts = df_valid_returns_trim.groupby(['frame_idx']).size().mean()
        df_return_counts = df_valid_returns_trim.groupby(['row', 'col']).size().reset_index().rename(columns={0:'count'})
        df_return_counts['count'] = df_return_counts['count']/100
        logger.info(df_return_counts)
        # Merge the count information back into the original dataframe
        # df_valid_returns_trim = pd.merge(df_valid_returns_trim, df_return_counts_trim, on=['Laser Shot Row', 'Laser Shot Cols'], how='left')
        
    elif sensor.sensor_type == "Hessai":
        df_valid_returns_trim = sensor.df_all_frames_target[['row', 'col', 'x' , 'y' , 'z', 'frame_idx']]
        # df_return_counts_trim = df_valid_returns_trim.groupby(['Laser Shot Row', 'Laser Shot Cols']).size().reset_index(name='count')
        # df_return_counts_trim['count'] = df_return_counts_trim['count'] / 100
        df_frame_counts = df_valid_returns_trim.groupby(['frame_idx']).size().mean()
        df_return_counts = df_valid_returns_trim.groupby(['row', 'col']).size().reset_index().rename(columns={0:'count'})
        df_return_counts['count'] = df_return_counts['count']/100

        
        # Merge the count information back into the original dataframe
        # df_valid_returns_trim = pd.merge(df_valid_returns_trim, df_return_counts_trim, on=['Laser Shot Row', 'Laser Shot Cols'], how='left')    
    
    logger.info(f' Data grouping complete!')
    return df_return_counts, df_frame_counts

def pd_calculation(sensor): #, df_return_counts, df_frame_counts

    df_return_counts, df_frame_counts = data_grouping(sensor)
    export_point_data(sensor, df_return_counts, folder_name="data_pd")
    logger.info(df_return_counts)
    if sensor.sensor_type == "voyant":
        max_points_on_target = df_return_counts['count'].count()
        max_y_range = df_return_counts['corrected_snr'].max() * 2
        avr_pd = df_return_counts['count'].mean()
        avr_range = sensor.df_all_frames_target['x'].mean()
        
        # Corrected SNR
        fig = sp.make_subplots(specs=[[{"secondary_y": True}]])
        # px.bar(df_return_counts, x = "point_idx", y = 'count', barmode="overlay")
        fig.add_trace(go.Bar(x= df_return_counts['point_idx'], y= df_return_counts['count'], name="average detections in 100 frames"), secondary_y=False)
        fig.add_trace(go.Bar(x= df_return_counts['point_idx'], y= df_return_counts['corrected_snr'], opacity= 0.8, name="linear SNR"), secondary_y=True)

        fig.update_yaxes(title_text="PD", secondary_y=False)
        fig.update_yaxes(title_text="SNR", secondary_y=True)
        fig.update_layout(
            legend=dict(orientation="h", ),
            yaxis2=dict(title='Linear SNR', side='right', overlaying='y', range=[0, max_y_range]),
            title=f'"Corrected" SNR based on 100 possible returns - Total number of points = {max_points_on_target}',
            )
        fig.update_xaxes(type='category')
        fig.update_yaxes(type='linear')
        fig.write_html(f"{sensor.directory}/PD_corLinSNR_{avr_pd:.3f}pd_{avr_range:.3f}m_graph.html")
        fig.show() #Might delete

        # MEAN SNR
        fig = sp.make_subplots(specs=[[{"secondary_y": True}]])
        # px.bar(df_return_counts, x = "point_idx", y = 'count', barmode="overlay")
        fig.add_trace(go.Bar(x= df_return_counts['point_idx'], y= df_return_counts['count'], name="average detections in 100 frames"), secondary_y=False)
        fig.add_trace(go.Bar(x= df_return_counts['point_idx'], y= df_return_counts['mean_snr'], opacity= 0.8, name="linear SNR"), secondary_y=True)

        max_y_range = df_return_counts['mean_snr'].max() * 2
        fig.update_yaxes(title_text="PD", secondary_y=False)
        fig.update_yaxes(title_text="SNR", secondary_y=True)
        fig.update_layout(
            legend=dict(orientation="h", ),
            yaxis2=dict(title='Linear SNR', side='right', overlaying='y', range=[0, max_y_range]),
            title=f'Mean SNR based on 100 possible returns - Total number of points = {max_points_on_target}',
            )
        fig.update_xaxes(type='category')
        fig.update_yaxes(type='linear')
        fig.write_html(f"{sensor.directory}/PD_meanLinSNR_{avr_pd:.3f}pd_{avr_range:.3f}m_graph.html")
        fig.show() #Might delete
 
    elif sensor.sensor_type == "scantinel":
        max_points_on_target = df_return_counts['count'].count()
        avr_pd: str = df_frame_counts
        avr_range = sensor.df_all_frames_target['x'].mean()

        # Plot
        fig = px.scatter(df_return_counts, x='col', y="row", color="count", width=800, height=800 ,title=f" average points = {avr_pd:.3f} at ~{avr_range:.3f}m")
        fig.write_html(f"{sensor.directory}/PD_{avr_pd:.3f}pd_{avr_range:.3f}m_graph.html")
        fig.show() #Might delete

    elif sensor.sensor_type == "silc": # this is new
        max_points_on_target = df_return_counts['count'].count()
        avr_pd: str = df_frame_counts
        avr_range = sensor.df_all_frames_target['y'].mean()

        # Plot
        fig = px.scatter(df_return_counts, x='col', y="row", color="count", width=800, height=800 ,title=f" average points = {avr_pd:.3f} at ~{avr_range:.3f}m")
        fig.write_html(f"{sensor.directory}/PD_{avr_pd:.3f}pd_{avr_range:.3f}m_graph.html")
        fig.show() #Might delete

        
    elif sensor.sensor_type == "HRL":
        logger.info(df_return_counts)
        max_points_on_target = df_return_counts['count'].count()
        logger.info(max_points_on_target)
        avr_pd: str = df_frame_counts
        avr_range = sensor.df_all_frames_target['x'].mean()
        
        # Plot
        fig = px.scatter(df_return_counts, x='Laser Shot Cols', y="Laser Shot Row", color="count", width=800, height=800 ,title=f" average points = {avr_pd:.3f} at ~{avr_range:.3f}m")
        #ig = px.scatter(df_return_counts, x='Laser Shot Cols', y="Laser Shot Row", color="count", width=800, height=800 ,title=f" average at ~{avr_range:.3f}m")
        #fig.write_html(f"{sensor.directory}/PD_{avr_pd:.3f}pd_{avr_range:.3f}m_graph.html")
        fig.write_html(f"{sensor.directory}/pd_{avr_range:.3f}m_graph.html")
        fig.show() #Might delete

    elif sensor.sensor_type == "SRL":
        max_points_on_target = df_return_counts['count'].count()
        avr_pd_og: str = df_frame_counts
        avr_pd_alt = round((sensor.df_selection['x'].count() + avr_pd_og)/2)
        
        if sensor.debug:
         ### When getting the stop sign reflection behind real stop sign 
         # return_X=sensor.df_all_frames_target[sensor.df_all_frames_target['return_idx']==0].copy()
         # avr_range=return_X['range'].mean()
         #Range given by a mean
         avr_range = sensor.df_all_frames_target['range'].mean()
         #In presence of aggressive outliers
         #avr_range = sensor.df_all_frames_target['range'].quantile(q=0.5)
         max_frames = int(sensor.df_all_frames_target['frame_idx'].max())
         logger.info(max_frames)
         high_pd_returns=df_return_counts[df_return_counts['count'] >= (max_frames*0.009)]
         #high_pd_returns=df_return_counts[df_return_counts['count']>=0.90]
         avr_pd= high_pd_returns['count'].count()
        else:
         ### When getting the stop sign reflection behind real stop sign 
         # return_X=sensor.df[sensor.df['return_idx']==0].copy()
         # avr_range=return_X['range'].mean()
         #Range given by a mean
         avr_range = sensor.df['range'].mean()
         #In presence of aggressive outliers
         #avr_range = sensor.df['range'].quantile(q=0.5)
         high_pd_returns=df_return_counts[df_return_counts['count']>=0.90]
         avr_pd= high_pd_returns['count'].count()
         logger.info(avr_range)
        
        # Plot
        fig = px.scatter(df_return_counts, x='col', y="row", color="count",color_continuous_scale=px.colors.sequential.Turbo, width=800, height=800 ,title=f" average points = {avr_pd_alt:.3f} at ~{avr_range:.3f}m with {avr_pd} points 90% PD")
        #ig = px.scatter(df_return_counts, x='Laser Shot Cols', y="Laser Shot Row", color="count", width=800, height=800 ,title=f" average at ~{avr_range:.3f}m")
        #fig.write_html(f"{sensor.directory}/PD_{avr_pd:.3f}pd_{avr_range:.3f}m_graph.html")
        fig.write_html(f"{sensor.directory}/pd_{avr_range:.3f}m_graph.html")
        fig.show() #Might delete
        

    elif sensor.sensor_type == "Hessai":
        # logger.info(df_return_counts)
        max_points_on_target = df_return_counts['count'].count()
        # logger.info(max_points_on_target)
        avr_pd: str = df_frame_counts
        avr_range = sensor.df_all_frames_target['x'].mean()
        
        # Plot
        fig = px.scatter(df_return_counts, x='col', y="row", color="count", width=800, height=800 ,title=f" average points = {avr_pd:.3f} at ~{avr_range:.3f}m")
        #ig = px.scatter(df_return_counts, x='Laser Shot Cols', y="Laser Shot Row", color="count", width=800, height=800 ,title=f" average at ~{avr_range:.3f}m")
        #fig.write_html(f"{sensor.directory}/PD_{avr_pd:.3f}pd_{avr_range:.3f}m_graph.html")
        fig.write_html(f"{sensor.directory}/pd_{avr_range:.3f}m_graph.html")
        fig.show() #Might delete

    
    logger.info(f' pd calculation complete, all results located in: {sensor.directory}!')
    return avr_pd

def pfa_calculation(sensor):

  sample = sensor.sensor_type

  if sample == "voyant":
        df_return_valid_count = sensor.df_valid_returns.groupby(['frame_idx']).size().reset_index().rename(columns={0:'count'})
        df_return_valid_count = df_return_valid_count.iloc[100:200] # This uses the second 100 frames... can use the first with [0,100]
        df_return_valid_count['count'] = df_return_valid_count['count']
        pfa = (df_return_valid_count['count'].mean() * 100)/sensor.shots_in_frame
        px.scatter(sensor.df_valid_returns, x='y', y='z', color='x', width=800, height=800, title=f"PFA = {pfa:.3f}").write_html(f"{sensor.directory}PFA_{pfa:.3f}_plot.html")

  elif sample == "scantinel":
        df_return_valid_count = sensor.df_valid_returns.groupby(['frame_idx']).size().reset_index().rename(columns={0:'count'})
        df_return_valid_count = df_return_valid_count.iloc[0:100] 
        df_return_valid_count['count'] = df_return_valid_count['count']
        pfa = (df_return_valid_count['count'].mean() * 100)/sensor.shots_in_frame
        px.scatter(sensor.df_valid_returns, x='y', y='z', color='x', width=800, height=800, title=f"PFA = {pfa:.3f}").write_html(f"{sensor.directory}PFA_{pfa:.3f}_plot.html")

  elif sample == "silc":
        df_return_valid_count = sensor.df_valid_returns.groupby(['frame_idx']).size().reset_index().rename(columns={0:'count'})
        df_return_valid_count = df_return_valid_count.iloc[0:100] 
        df_return_valid_count['count'] = df_return_valid_count['count']
        pfa = (df_return_valid_count['count'].mean() * 100)/sensor.shots_in_frame
        px.scatter(sensor.df_valid_returns, x='y', y='z', color='x', width=800, height=800, title=f"PFA = {pfa:.3f}").write_html(f"{sensor.directory}PFA_{pfa:.3f}_plot.html")

  elif sample == "HRL":
        df_return_valid_count = sensor.df_valid_returns.groupby(['frame_idx']).size().reset_index().rename(columns={0:'count'})
        df_return_valid_count = df_return_valid_count.iloc[0:100] 
        df_return_valid_count['count'] = df_return_valid_count['count']
        pfa = (df_return_valid_count['count'].mean() * 100)/sensor.shots_in_frame
        px.scatter(sensor.df_valid_returns, x='y', y='z', color='x', width=800, height=800, title=f"PFA = {pfa:.3f}").write_html(f"{sensor.directory}PFA_{pfa:.3f}_plot.html")
  elif sample == "SRL":
        df_return_valid_count = sensor.df_valid_returns.groupby(['frame_idx']).size().reset_index().rename(columns={0:'count'})
        df_return_valid_count = df_return_valid_count.iloc[0:100] 
        df_return_valid_count['count'] = df_return_valid_count['count']
        pfa = (df_return_valid_count['count'].mean() * 100)/sensor.shots_in_frame
        px.scatter(sensor.df_valid_returns, x='y', y='z', color='x', width=800, height=800, title=f"PFA = {pfa:.3f}").write_html(f"{sensor.directory}PFA_{pfa:.3f}_plot.html")
  elif sample == "Hessai":
        df_return_valid_count = sensor.df_valid_returns.groupby(['frame_idx']).size().reset_index().rename(columns={0:'count'})
        df_return_valid_count = df_return_valid_count.iloc[0:100] 
        df_return_valid_count['count'] = df_return_valid_count['count']
        pfa = (df_return_valid_count['count'].mean() * 100)/sensor.shots_in_frame
        px.scatter(sensor.df_valid_returns, x='y', y='z', color='x', width=800, height=800, title=f"PFA = {pfa:.3f}").write_html(f"{sensor.directory}PFA_{pfa:.3f}_plot.html")

  export_point_data(sensor, df_return_valid_count, folder_name="data_PFA")
  logger.info(f' PFA calculation complete!')
  return df_return_valid_count, pfa

# %%

def range_precision(sensor):
    if sensor.sensor_type == "voyant":
        df_valid_returns_trim = sensor.df_all_frames_target[['point_idx','x' , 'y' , 'z', 'range']]
        df_return_precision = df_valid_returns_trim.groupby(['point_idx']).std(ddof=0)['range'].reset_index().rename(columns={0:'stdv_x'})

    elif sensor.sensor_type == "scantinel": # this is new
        df_valid_returns_trim = sensor.df_all_frames_target[['col', 'row', 'x' , 'y' , 'z','range']]
        df_return_precision = df_valid_returns_trim.groupby(['col', 'row']).std(ddof=0)['range'].reset_index().rename(columns={0:'stdv_x'})

    elif sensor.sensor_type == "silc": # this is new
        df_valid_returns_trim = sensor.df_all_frames_target[['col', 'row', 'x' , 'y' , 'z','range']]
        df_return_precision = df_valid_returns_trim.groupby(['col', 'row']).std(ddof=0)['range'].reset_index().rename(columns={0:'stdv_x'})
       

    elif sensor.sensor_type == "HRL": # this is new
        df_valid_returns_trim = sensor.df_all_frames_target[['Laser Shot Cols', 'Laser Shot Row', 'x' , 'y' , 'z','range']]
        df_return_precision = df_valid_returns_trim.groupby(['Laser Shot Cols', 'Laser Shot Row']).std(ddof=0)['range'].reset_index().rename(columns={0:'stdv_x'})
        # df_return_counts['count'] = df_return_counts['count']/100

    elif sensor.sensor_type == "SRL": # this is new
        df_valid_returns_trim = sensor.df_all_frames_target[['col', 'row', 'x' , 'y' , 'z','range']]
        df_return_precision = df_valid_returns_trim.groupby(['col', 'row']).std(ddof=0)['range'].reset_index().rename(columns={0:'stdv_x'})
        # df_return_counts['count'] = df_return_counts['count']/100

    elif sensor.sensor_type == "Hessai": # this is new
        df_valid_returns_trim = sensor.df_all_frames_target[['col', 'row', 'x' , 'y' , 'z','range']]
        df_return_precision = df_valid_returns_trim.groupby(['col', 'row']).std(ddof=0)['range'].reset_index().rename(columns={0:'stdv_x'})
        # df_return_counts['count'] = df_return_counts['count']/
    
    average_std = df_return_precision['range'].mean()

    # df_return_precision.to_csv(f"{filepath}{data_name}_range_prec_data.txt", sep=',')
    export_point_data(sensor,df_return_precision,folder_name="data_range_precision")

    logger.info(f'Range precision complete!')
    return df_return_precision, average_std

def target_size(sensor):
    if sensor.sensor_type == "voyany":
     pass

    elif sensor.sensor_type =="scantinel":
     pass

    elif sensor.sensor_type == "silc":
     pass

    elif sensor.sensor_type == "HRL":
     z_min=sensor.df_selection['z'].min()
     z_max=sensor.df_selection['z'].max()
     y_min=sensor.df_selection['y'].min()
     y_max=sensor.df_selection['y'].max()
     TargHeight=z_max-z_min
     TargWidth=y_max-y_min
     df_2d_target=sensor.df_selection[['y', 'z']] 
     hull=ConvexHull(df_2d_target) #hull contains the the graham scan points from the given array
     _ = convex_hull_plot_2d(hull)
     plt.show()
     TargPerimeter=hull.area
     TargArea=hull.volume
     Targ3d= px.scatter_3d(sensor.df_selection, x='x', y='y', z='z',width=800, height=800, title="Target")
     Targ3d.show()
     Targ2d= px.scatter(sensor.df_selection,x='y',y='z', width=800, height=800,title="Target Graph")
     Targ2d.show()
     logger.info(f'Target Size calculation complete')

    elif sensor.sensor_type == "SRL":
     z_min=sensor.df_selection['z'].min()
     z_max=sensor.df_selection['z'].max()
     y_min=sensor.df_selection['y'].min()
     y_max=sensor.df_selection['y'].max()
     TargHeight=z_max-z_min
     TargWidth=y_max-y_min
     df_2d_target=sensor.df_selection[['y', 'z']] 
     hull=ConvexHull(df_2d_target) #hull contains the the graham scan points from the given array
     _ = convex_hull_plot_2d(hull)
     plt.show()
     TargPerimeter=hull.area
     TargArea=hull.volume
     Targ3d= px.scatter_3d(sensor.df_selection, x='x', y='y', z='z',width=800, height=800, title="Target")
     Targ3d.show()
     Targ2d= px.scatter(sensor.df_selection,x='y',y='z', width=800, height=800,title=f"Target Size {TargHeight:.3f} by ~{TargWidth:.3f}m")
     Targ2d.show()
     logger.info(f'Target Size calculation complete')

    elif sensor.sensor_type == "Hessai":
     z_min=sensor.df_selection['z'].min()
     z_max=sensor.df_selection['z'].max()
     y_min=sensor.df_selection['y'].min()
     y_max=sensor.df_selection['y'].max()
     TargHeight=z_max-z_min
     TargWidth=y_max-y_min
     df_2d_target=sensor.df_selection[['y', 'z']] 
     hull=ConvexHull(df_2d_target) #hull contains the the graham scan points from the given array
     _ = convex_hull_plot_2d(hull)
     plt.show()
     TargPerimeter=hull.area
     TargArea=hull.volume
     Targ3d= px.scatter_3d(sensor.df_selection, x='x', y='y', z='z',width=800, height=800,)
     Targ3d.show()
     Targ2d= px.scatter(sensor.df_selection,x='y',y='z', width=800, height=800,)
     Targ2d.write_image('TargetSize_',TargHeight,'m',TargWidth,'m','.png')
     Targ2d.show()
     logger.info(f'Target Size calculation complete')

    return TargHeight, TargWidth, TargPerimeter, TargArea
   
def get_sensor_and_target_types(sensor):
    # Descr: Assign the input data path to 'directory'
    # Input: sample_type and target_type.
    # Output: 'directory'
    # Needs improved user input error checking.
    # This code block was written by Bruce.

    # Query user to select which sample_type to analyze.
    sensor_list = ['ars542', 'HRL', 'scantinel', 'silc', 'voyant']
    nn = 0
    for sensor_n in sensor_list:
        print(f"{nn}: {sensor_n}")
        nn += 1

    print('Enter the number of the sample_type to analyze (default is silc).')
    nn = int(input() or "3")
    sensor.sensor_type = sensor_list[nn]
    print(f"Analyzing: {sensor.sensor_type}\n")

    # Query user to select which target to analyze.
    target_list = ['10P', 'STOP']
    print('0: 10P\n1: STOP')
    print('Enter the number of the target_type to analyze (default is 10%).')
    nn = int(input() or "0")
    target_type = target_list[nn]
    print(f"Target selected: {target_type}\n")
    return target_type

def localization_filter(df_all_frames, sensor, nstd=1):
    # Descr: Apply a localization-filter by removing points which lie outside of 
    # x- , y-, and z-limits.
    # Input: 'df_all_frames'
    # Output: 'df_all_frames_filtered'

    # Check localization filter-width-multiplifcation-factor (nstd). 
    # Get user input. Repeat until it is valid. 'nstd' is in units of std's
    while nstd<=0 or nstd>100:
        print(f"Enter the Localization filter-width-multiplication-factor. Must be > 0 and <100.")
        nstd = float(input())
        print(f'Value entered: {nstd}')
        if nstd>0 and nstd<100:
            break

    tempdata = []   # empty list for storing single dataframes
    df_all_frames_stats = compute_statistics(df_all_frames, sensor)

    for frame_nr in df_all_frames_stats['frame_idx'].unique():
        current_row_stats = df_all_frames_stats[df_all_frames_stats['frame_idx']==frame_nr]
        if current_row_stats['count'].item() < 10:          # Skip/remove frames w/ count < 10
            continue
        MIN_X = current_row_stats['x_mean'].item() - nstd*current_row_stats['x_std'].item()
        MAX_X = current_row_stats['x_mean'].item() + nstd*current_row_stats['x_std'].item()
        MIN_Y = current_row_stats['y_mean'].item() - nstd*current_row_stats['y_std'].item()
        MAX_Y = current_row_stats['y_mean'].item() + nstd*current_row_stats['y_std'].item()
        MIN_Z = current_row_stats['z_mean'].item() - nstd*current_row_stats['z_std'].item()
        MAX_Z = current_row_stats['z_mean'].item() + nstd*current_row_stats['z_std'].item()

        df_singleframe = df_all_frames[df_all_frames['frame_idx'] == frame_nr]
        df_singleframe_lfiltered = df_singleframe[(df_singleframe['x'] < MAX_X)
                                        & (df_singleframe['x'] >= MIN_X)
                                        & (df_singleframe['y'] < MAX_Y)
                                        & (df_singleframe['y'] >= MIN_Y)
                                        & (df_singleframe['z'] < MAX_Z)
                                        & (df_singleframe['z'] >= MIN_Z)
                                        ]
        tempdata.append(df_singleframe_lfiltered)

    df_all_frames_filtered = pd.concat(tempdata)
    return df_all_frames_filtered

def compute_statistics(df_all_frames, sensor):
    # Descr: compute statistics (min,max,mean,std,count) for Velocity-Analysis.
    # Input: 'df_all_frames'
    # Output: 'df_all_frames_stats'
    # This code block as written by Danny.

    if sensor.sensor_type == "ars542":
        # Rename the f_vrelRad column to radial_vel
        df_all_frames.rename(columns={
            'f_vrelRad': 'radial_vel',
        }, inplace=True)

        df_all_frames_stats = df_all_frames.groupby('frame_idx').agg({
            'x': ['min', 'mean', 'max', 'std', 'count'],
            'y': ['min', 'mean', 'max', 'std',],
            'z': ['min', 'mean', 'max', 'std',],
            'f_rangeRad': ['min', 'mean', 'max', 'std',],
            'radial_vel': ['min', 'mean', 'max', 'std',],
            'f_azAng': ['min', 'mean', 'max', 'std',],
            'f_elevAng': ['min', 'mean', 'max', 'std',],
            'f_RCS': ['min', 'mean', 'max', 'std',],
            'f_SNR': ['min', 'mean', 'max', 'std',],
            'u_vrelAmbigResolved': ['min', 'mean', 'max', 'std',],
        })
    elif sensor.sensor_type == "HRL":
       pass

    elif sensor.sensor_type =="scantinel":
        # Rename the Velocity column to radial_vel
        df_all_frames.rename(columns={
            'Velocity': 'radial_vel',
        }, inplace=True)

        df_all_frames_stats = df_all_frames.groupby('frame_idx').agg({
            'x': ['min', 'mean', 'max', 'std', 'count'],
            'y': ['min', 'mean', 'max', 'std',],
            'z': ['min', 'mean', 'max', 'std',],
            'Range': ['min', 'mean', 'max', 'std',],
            'Intensity': ['min', 'mean', 'max', 'std',],
            'Scaled Intensity': ['min', 'mean', 'max', 'std',],
            'radial_vel': ['min', 'mean', 'max', 'std',],
        })

    elif sensor.sensor_type == "silc":
        # Rename the velocity column to radial_vel
        df_all_frames.rename(columns={
            'velocity': 'radial_vel',
        }, inplace=True)

        df_all_frames_stats = df_all_frames.groupby('frame_idx').agg({
            'x': ['min', 'mean', 'max', 'std', 'count'],
            'y': ['min', 'mean', 'max', 'std',],
            'z': ['min', 'mean', 'max', 'std',],
            'power_te': ['min', 'mean', 'max', 'std',],
            'power_tm': ['min', 'mean', 'max', 'std',],
            'radial_vel': ['min', 'mean', 'max', 'std',],
            'azimuth_angle': ['min', 'mean', 'max', 'std'],
            'elevation_angle': ['min', 'mean', 'max', 'std'],
        })

    elif sensor.sensor_type == "voyant":
        # Rename the velocity column to radial_vel
        df_all_frames.rename(columns={
            'velocity': 'radial_vel',
        }, inplace=True)
        
        df_all_frames_stats = df_all_frames.groupby('frame_idx').agg({
            'x': ['min', 'mean', 'max', 'std', 'count'],
            'y': ['min', 'mean', 'max', 'std',],
            'z': ['min', 'mean', 'max', 'std',],
            'radial_vel': ['min', 'mean', 'max', 'std',],
            'snr_linear': ['min', 'mean', 'max', 'std',],
        })

    # Flatten the multi-index columns
    df_all_frames_stats.columns = ['_'.join(col) for col in df_all_frames_stats.columns]

    # Rename the count columns to represent count
    df_all_frames_stats.rename(columns={
        'x_count': 'count',
    }, inplace=True)

    # Reset index to have frame_idx as a column
    df_all_frames_stats.reset_index(inplace=True)

    # Add a new column (radial_vel_percent) derived from existing columns (radial_vel_std, radial_vel_mean)
    df_all_frames_stats['radial_vel_percent'] = 100*df_all_frames_stats['radial_vel_std']/df_all_frames_stats['radial_vel_mean']

    return df_all_frames_stats

def make_velocity_histogram(sensor, target_type, selected_velocity, tolerance):
    # Inputs
    # this is meters per seconds not miles per hour
    # tolerance = +/- selected_velocity value
    # 2 for SiLC & Voyant
    vel_min_boundary = selected_velocity - tolerance
    vel_max_boundary = selected_velocity + tolerance

    sample_text = sensor.sensor_type + ', ' +  target_type
    
    if sensor.sensor_type == "ars542":
        fig = px.histogram(sensor.df['f_vrelRad'], log_y=True)
        fig.update_layout(
            xaxis_title="Radial Velocity [m/s]",
            yaxis_title="Counts",
            title_text="Velocity Histogram",
            font=dict(size=16),
        )
        fig.add_annotation(text=sample_text,
            xref="x domain",
            yref="y domain",
            x=0.01, y=1, showarrow=False,
            font=dict(color="red"),
            )
        fig.add_vline(x=vel_min_boundary, line_width=2, line_dash='dash', line_color='red')
        fig.add_vline(x=vel_max_boundary, line_width=2, line_dash='dash', line_color='red')
        fig.write_html(f"{sensor.sensor_type}_{target_type}_f_vhistogram.html")    
        fig.show()

    elif sensor.sensor_type == "HRL":
       pass

    elif sensor.sensor_type =="scantinel":
        fig = px.histogram(sensor.df['Velocity'], log_y=True)
        fig.update_layout(
            xaxis_title="Radial Velocity [m/s]",
            yaxis_title="Counts",
            title_text="Velocity Histogram",
            font=dict(size=16),
        )
        fig.add_annotation(text=sample_text,
            xref="x domain",
            yref="y domain",
            x=0.01, y=1, showarrow=False,
            font=dict(color="red"),
            )
        fig.add_vline(x=vel_min_boundary, line_width=2, line_dash='dash', line_color='red')
        fig.add_vline(x=vel_max_boundary, line_width=2, line_dash='dash', line_color='red')
        fig.show()
        fig.write_html(f"{sensor.sensor_type}_{target_type}_f_vhistogram.html")    

    elif sensor.sensor_type == "silc":
        fig = px.histogram(sensor.df['velocity'], log_y=True)
        fig.update_layout(
            xaxis_title="Radial Velocity [m/s]",
            yaxis_title="Counts",
            title_text="Velocity Histogram",
            font=dict(size=16),
        )
        fig.add_annotation(text=sample_text,
            xref="x domain",
            yref="y domain",
            x=0.01, y=1, showarrow=False,
            font=dict(color="red"),
            )
        fig.add_vline(x=vel_min_boundary, line_width=2, line_dash='dash', line_color='red')
        fig.add_vline(x=vel_max_boundary, line_width=2, line_dash='dash', line_color='red')
        fig.show()
        fig.write_html(f"{sensor.sensor_type}_{target_type}_f_vhistogram.html")    

    elif sensor.sensor_type == "voyant":
        fig = px.histogram(sensor.df['radial_vel'], log_y=True)
        fig.update_layout(
            xaxis_title="Radial Velocity [m/s]",
            yaxis_title="Counts",
            title_text="Velocity Histogram",
            font=dict(size=16),
        )
        fig.add_annotation(text=sample_text,
            xref="x domain",
            yref="y domain",
            x=0.01, y=1, showarrow=False,
            font=dict(color="red"),
            )
        fig.add_vline(x=vel_min_boundary, line_width=2, line_dash='dash', line_color='red')
        fig.add_vline(x=vel_max_boundary, line_width=2, line_dash='dash', line_color='red')
        fig.show()
        fig.write_html(f"{sensor.sensor_type}_{target_type}_f_vhistogram.html")

def make_velocity_plots(df_all_frames, sensor, target_type, threshold_count=2):
    # Descr: Generate Velocity Analysis results plots

    # 'threshold_count' is used to remove frames which have 'count' below 'threshold_count', 
    # and for computing 'radial_vel_percent'.mean().

    sample_text = sensor.sensor_type + ', ' + target_type

    # Make plots which use points-data
    fig = px.scatter(df_all_frames, x='x', y='y', color='frame_idx')
    fig.update_layout(
        xaxis_title="Longitudinal Distance [m] (X-axis)",
        yaxis_title="Lateral Distance [m] (Y-axis)",
        title_text="Distance",
        font=dict(size=16),
    )
    fig.add_annotation(text=sample_text,
        xref="x domain",
        yref="y domain",
        x=0.01, y=1, showarrow=False,
        font=dict(color="red"),
        )
    fig.show()
    fig.write_html(f"1_{sensor.sensor_type}_{target_type}_ctf_distance_vs_frame.html")

    df_all_frames_stats = compute_statistics(df_all_frames, sensor)

    # Homogenize the data for different sensor types
    if sensor.sensor_type == "ars542":
        df_all_frames_stats['range_mean'] = df_all_frames_stats['f_rangeRad_mean']
    elif sensor.sensor_type == "HRL":
        pass
    elif sensor.sensor_type =="scantinel":
        df_all_frames_stats['range_mean'] = df_all_frames_stats['Range_mean']
    elif sensor.sensor_type == "silc":
        df_all_frames_stats['range_mean'] = (df_all_frames_stats['x_mean']**2 + df_all_frames_stats['y_mean']**2 + df_all_frames_stats['z_mean']**2)**.5
    elif sensor.sensor_type == "voyant":
        df_all_frames_stats['range_mean'] = (df_all_frames_stats['x_mean']**2 + df_all_frames_stats['y_mean']**2 + df_all_frames_stats['z_mean']**2)**.5


    # Make plots which use modified stats-data
    # Remove frames which have 'counts' below threshold. Then plot and output 'radial_vel_percent' KPI's
    df_all_frames_stats_valid = df_all_frames_stats[df_all_frames_stats['count']>=threshold_count]
    df_all_frames_stats_valid['radial_vel_percent'] = df_all_frames_stats_valid['radial_vel_percent'].abs()

    fig = px.line(df_all_frames_stats_valid, x='range_mean', y='count', log_y=True)
    fig.update_layout(
        xaxis_title="Range_mean [m]",
        yaxis_title="Number of Points (filtered)",
        title_text="Number of Points",
        font=dict(size=16),
        )
    fig.add_annotation(text=sample_text,
        xref="x domain",
        yref="y domain",
        x=0.01, y=1, showarrow=False,
        font=dict(color="red"),
        )
    fig.add_hline(y=threshold_count, line_width=2, line_dash='dash', line_color='red')
    fig.show()
    fig.write_html(f"2_{sensor.sensor_type}_{target_type}_ctf_count_vs_range.html")

    fig = go.Figure(data=go.Scatter(
        # x=df_all_frames_stats['frame_idx'],
        x=df_all_frames_stats_valid['range_mean'],
        y=df_all_frames_stats_valid['radial_vel_mean'].abs(),
        error_y=dict(
            type='data',
            array=df_all_frames_stats_valid['radial_vel_std'],
            visible=True)
    ))
    fig.update_layout(
        # xaxis_title="Time (frame_idx)",
        xaxis_title="Range_mean [m]",
        yaxis_title="Radial Velocity_mean_abs [m/s]",
        title_text="Radial Velocity_mean",
        font=dict(size=16),
        )
    fig.add_annotation(text=sample_text,
        xref="x domain",
        yref="y domain",
        x=0.01, y=1, showarrow=False,
        font=dict(color="red"),
        )
    fig.show()
    fig.write_html(f"3_{sensor.sensor_type}_{target_type}_ctf_radvel_vs_range.html")
    # --------

    # fig = px.line(df_all_frames_stats_valid, x='frame_idx', y='radial_vel_percent')
    fig = px.line(df_all_frames_stats_valid, x='range_mean', y='radial_vel_percent')
    fig.update_layout(
        # xaxis_title="Time (frame_idx)",
        xaxis_title="Range_mean [m]",
        yaxis_title="Radial Velocity Noise_abs [%]",
        title_text="Radial Velocity Noise",
        font=dict(size=16),
    )
    fig.add_annotation(text=sample_text,
        xref="x domain",
        yref="y domain",
        x=0.01, y=1, showarrow=False,
        font=dict(color="red"),
        )
    value_str = "radial_vel_noise = " + "%.2f"% df_all_frames_stats_valid['radial_vel_percent'].mean() + "%"
    fig.add_annotation(text=value_str,
        xref="x domain",
        yref="y domain",
        x=0.01, y=0.9, showarrow=False,
        font=dict(color="red"),
        )
    fig.show()
    fig.write_html(f"4_{sensor.sensor_type}_{target_type}_ctf_radvelpercent_vs_range.html")

def make_hrl_velocity_plots(df_all_frames, df_all_frames_stats, sensor, target_type):
    if sensor.sensor_type == "ars542":
       pass

    elif sensor.sensor_type =="scantinel":
        pass

    elif sensor.sensor_type == "silc":
        pass

    elif sensor.sensor_type == "voyant":
        pass

    elif sensor.sensor_type == "HRL":
        sample_text = sensor.sensor_type + ', ' + target_type

        fig = px.scatter(df_all_frames, x='x', y='y', color='frame_idx')
        fig.update_layout(
            xaxis_title="Longitudinal Distance [m] (X-axis)",
            yaxis_title="Lateral Distance [m] (Y-axis)",
            title_text="Distance",
            font=dict(size=16),
        )
        fig.add_annotation(text=sample_text,
            xref="x domain",
            yref="y domain",
            x=0.01, y=1, showarrow=False,
            font=dict(color="red"),
            )
        fig.show()
        fig.write_html(f"1_{sensor.sensor_type}_{target_type}_ctf_distance_vs_frame.html")
    
        df_all_frames_stats.reset_index(inplace=True)
        df_all_frames_stats['velocity_abs']=df_all_frames_stats['velocity'].abs()

        # Flatten the multi-index columns
        df_all_frames_stats.columns = ['_'.join(col) for col in df_all_frames_stats.columns]

        fig = px.line(df_all_frames_stats, x='range_', y='x_count', log_y=True)
        fig.update_layout(
            xaxis_title="Range [m]",
            yaxis_title="Number of Points (filtered)",
            title_text="HRL Number of Points",
            font=dict(size=16),
            )
        fig.add_annotation(text=sample_text,
            xref="x domain",
            yref="y domain",
            x=0.01, y=1, showarrow=False,
            font=dict(color="red"),
            )
        fig.show()
        fig.write_html(f"2_{sensor.sensor_type}_{target_type}_ctf_count_vs_range.html")

        fig = px.line(df_all_frames_stats, x='range_', y='velocity_abs_', log_y=False)
        fig.update_layout(
            xaxis_title="Range [m]",
            yaxis_title="Radial Velocity_calculated_abs [m/s]",
            title_text="HRL Velocity",
            font=dict(size=16),
            )
        fig.add_annotation(text=sample_text,
            xref="x domain",
            yref="y domain",
            x=0.01, y=1, showarrow=False,
            font=dict(color="red"),
            )
        fig.show()
        fig.write_html(f"3_{sensor.sensor_type}_{target_type}_ctf_radvelcalc_vs_range.html")

def make_multisensor_velocity_plot(df_stats_ars, df_stats_hrl, df_stats_scantinel, df_stats_silc, target_type, target_speed):
    # Make plot of Multi-sensor data overlaid on one plot axes
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_stats_ars['f_rangeRad_mean'],
        y=df_stats_ars['radial_vel_mean'].abs(),
        error_y=dict(
            type='data',
            array=df_stats_ars['radial_vel_std'],
            visible=True),
        mode='lines',
        name='ars_ctf'))
    fig.add_trace(go.Scatter(
        x=df_stats_scantinel['Range_mean'],
        y=df_stats_scantinel['radial_vel_mean'].abs(),
        error_y=dict(
            type='data',
            array=df_stats_scantinel['radial_vel_std'],
            visible=True),
        mode='lines',
        name='scantinel_ctf'))
    fig.add_trace(go.Scatter(
        x=df_stats_silc['range_mean'],
        y=df_stats_silc['radial_vel_mean'].abs(),
        error_y=dict(
            type='data',
            array=df_stats_silc['radial_vel_std'],
            visible=True),
        mode='lines',
        name='silc_ctf'))
    fig.add_trace(go.Scatter(
        x=df_stats_hrl['range_'],
        y=df_stats_hrl['velocity_'].abs(),
        mode='lines',
        name='hrl_ctf'))
    fig.update_layout(
        xaxis_title="Range_mean [m]",
        yaxis_title="Radial Velocity_mean_abs [m/s]",
        title_text="Multi-sensor, Radial Velocity, 10%, " + target_speed,
        font=dict(size=16),
        )
    fig.update_layout(yaxis_range=[8, 12])
    fig.show()
    fig.write_html(f"Multi-sensor_{target_type}_{target_speed}_ctf_radvel_mean_abs_vs_range.html")
    

def intensity_distribution(df_silc_80p_10m, df_silc_80p_20m, df_silc_80p_50m):
    # Example data
    x0 = df_silc_80p_10m['power_te']
    x1 = df_silc_80p_20m['power_te']
    x2 = df_silc_80p_50m['power_te']

    # Create a dataframe
    df = pd.DataFrame({
        'series': np.concatenate((["10m"] * len(x0), ["20m"] * len(x1), ["50m"] * len(x2))), 
        'data': np.concatenate((x0, x1, x2)),
    })

    # Plot the histogram
    fig = px.histogram(df, x="data", color="series", barmode="overlay")

    # Update layout with title and labels
    fig.update_layout(
        title="SiLC Intensity Distribution over 100 Frames (All Points)",
        xaxis_title="Total Power (pW)",
        yaxis_title="Count",
            annotations=[
            dict(
                text=f'Sensor: SILC<br>Target Type: 80%<br>Points: All<br>Frames: 100',
                x=0,
                y=1,
                xref='paper',
                yref='paper',
                xanchor='left',
                yanchor='top',
                showarrow=False,
                font=dict(
                    size=12,
                    color='black'
                ),
                bgcolor='rgba(255, 255, 255, 0.7)',
                bordercolor='rgba(0, 0, 0, 0.7)',
                borderwidth=1,
                align='left',
                borderpad=4,
                xshift=10,
                yshift=-10
            )
        ]
    )

    # Show the plot
    fig.show()



def export_point_data(sensor,data_frame, folder_name="test"):
        try:
            # Path 
            folder_path = os.path.join(sensor.directory, folder_name) 
            os.mkdir(folder_path)
        except:
            logger.warning("Error making folder and exporting data!")
   
def get_sensor_and_target_types(sensor):
    # Descr: Assign the input data path to 'directory'
    # Input: sample_type and target_type.
    # Output: 'directory'
    # Needs improved user input error checking.
    # This code block was written by Bruce.

    # Query user to select which sample_type to analyze.
    sensor_list = ['ars542', 'HRL', 'scantinel', 'silc', 'voyant']
    nn = 0
    for sensor_n in sensor_list:
        print(f"{nn}: {sensor_n}")
        nn += 1

    print('Enter the number of the sample_type to analyze (default is silc).')
    nn = int(input() or "3")
    sensor.sensor_type = sensor_list[nn]
    print(f"Analyzing: {sensor.sensor_type}\n")

    # Query user to select which target to analyze.
    target_list = ['10P', 'STOP']
    print('0: 10P\n1: STOP')
    print('Enter the number of the target_type to analyze (default is 10%).')
    nn = int(input() or "0")
    target_type = target_list[nn]
    print(f"Target selected: {target_type}\n")
    return target_type

def localization_filter(df_all_frames, sensor, nstd=1):
    # Descr: Apply a localization-filter by removing points which lie outside of 
    # x- , y-, and z-limits.
    # Input: 'df_all_frames'
    # Output: 'df_all_frames_filtered'

    # Check localization filter-width-multiplifcation-factor (nstd). 
    # Get user input. Repeat until it is valid. 'nstd' is in units of std's
    while nstd<=0 or nstd>100:
        print(f"Enter the Localization filter-width-multiplication-factor. Must be > 0 and <100.")
        nstd = float(input())
        print(f'Value entered: {nstd}')
        if nstd>0 and nstd<100:
            break

    tempdata = []   # empty list for storing single dataframes
    df_all_frames_stats = compute_statistics(df_all_frames, sensor)

    for frame_nr in df_all_frames_stats['frame_idx'].unique():
        current_row_stats = df_all_frames_stats[df_all_frames_stats['frame_idx']==frame_nr]
        if current_row_stats['count'].item() < 10:          # Skip/remove frames w/ count < 10
            continue
        MIN_X = current_row_stats['x_mean'].item() - nstd*current_row_stats['x_std'].item()
        MAX_X = current_row_stats['x_mean'].item() + nstd*current_row_stats['x_std'].item()
        MIN_Y = current_row_stats['y_mean'].item() - nstd*current_row_stats['y_std'].item()
        MAX_Y = current_row_stats['y_mean'].item() + nstd*current_row_stats['y_std'].item()
        MIN_Z = current_row_stats['z_mean'].item() - nstd*current_row_stats['z_std'].item()
        MAX_Z = current_row_stats['z_mean'].item() + nstd*current_row_stats['z_std'].item()

        df_singleframe = df_all_frames[df_all_frames['frame_idx'] == frame_nr]
        df_singleframe_lfiltered = df_singleframe[(df_singleframe['x'] < MAX_X)
                                        & (df_singleframe['x'] >= MIN_X)
                                        & (df_singleframe['y'] < MAX_Y)
                                        & (df_singleframe['y'] >= MIN_Y)
                                        & (df_singleframe['z'] < MAX_Z)
                                        & (df_singleframe['z'] >= MIN_Z)
                                        ]
        tempdata.append(df_singleframe_lfiltered)

    df_all_frames_filtered = pd.concat(tempdata)
    return df_all_frames_filtered

def compute_statistics(df_all_frames, sensor):
    # Descr: compute statistics (min,max,mean,std,count) for Velocity-Analysis.
    # Input: 'df_all_frames'
    # Output: 'df_all_frames_stats'
    # This code block as written by Danny.

    if sensor.sensor_type == "ars542":
        df_all_frames_stats = df_all_frames.groupby('frame_idx').agg({
            'x': ['min', 'mean', 'max', 'std', 'count'],
            'y': ['min', 'mean', 'max', 'std',],
            'z': ['min', 'mean', 'max', 'std',],
            'range': ['min', 'mean', 'max', 'std',],
            'velocity': ['min', 'mean', 'max', 'std',],
            'f_azAng': ['min', 'mean', 'max', 'std',],
            'f_elevAng': ['min', 'mean', 'max', 'std',],
            'f_RCS': ['min', 'mean', 'max', 'std',],
            'f_SNR': ['min', 'mean', 'max', 'std',],
            'u_vrelAmbigResolved': ['min', 'mean', 'max', 'std',],
        })
    elif sensor.sensor_type == "HRL":
       pass

    elif sensor.sensor_type =="scantinel":
        df_all_frames_stats = df_all_frames.groupby('frame_idx').agg({
            'x': ['min', 'mean', 'max', 'std', 'count'],
            'y': ['min', 'mean', 'max', 'std',],
            'z': ['min', 'mean', 'max', 'std',],
            'range': ['min', 'mean', 'max', 'std',],
            'velocity': ['min', 'mean', 'max', 'std',],
            'Intensity': ['min', 'mean', 'max', 'std',],
            'Scaled Intensity': ['min', 'mean', 'max', 'std',],
        })

    elif sensor.sensor_type == "silc":
        df_all_frames_stats = df_all_frames.groupby('frame_idx').agg({
            'x': ['min', 'mean', 'max', 'std', 'count'],
            'y': ['min', 'mean', 'max', 'std',],
            'z': ['min', 'mean', 'max', 'std',],
            'range': ['min', 'mean', 'max', 'std',],
            'velocity': ['min', 'mean', 'max', 'std',],
            'power_te': ['min', 'mean', 'max', 'std',],
            'power_tm': ['min', 'mean', 'max', 'std',],
            'azimuth_angle': ['min', 'mean', 'max', 'std'],
            'elevation_angle': ['min', 'mean', 'max', 'std'],
        })

    elif sensor.sensor_type == "voyant":
        df_all_frames_stats = df_all_frames.groupby('frame_idx').agg({
            'x': ['min', 'mean', 'max', 'std', 'count'],
            'y': ['min', 'mean', 'max', 'std',],
            'z': ['min', 'mean', 'max', 'std',],
            'range': ['min', 'mean', 'max', 'std',],
            'velocity': ['min', 'mean', 'max', 'std',],
            'snr_linear': ['min', 'mean', 'max', 'std',],
        })

    # Flatten the multi-index columns
    df_all_frames_stats.columns = ['_'.join(col) for col in df_all_frames_stats.columns]

    # Rename the count columns to represent count
    df_all_frames_stats.rename(columns={
        'x_count': 'count',
    }, inplace=True)

    # Reset index to have frame_idx as a column
    df_all_frames_stats.reset_index(inplace=True)

    # Add a new column (velocity_percent) derived from existing columns (velocity_std, velocity_mean)
    df_all_frames_stats['velocity_percent'] = 100*df_all_frames_stats['velocity_std']/df_all_frames_stats['velocity_mean']

    return df_all_frames_stats

def make_velocity_histogram(sensor, target_type, target_speed, selected_velocity=0, tolerance=0):
    # Inputs
    # this is meters per seconds not miles per hour
    # tolerance = +/- selected_velocity value
    # 2 for SiLC & Voyant

    if sensor.sensor_type != ('ars542' or 'scantinel' or 'silc' or 'voyant'):
        return          # return for HRL and other sensor_types
    
    sample_text = sensor.sensor_type + ', ' + target_type + ', ' + target_speed
    
    # Compute parameters for plotting
    nbins = round(1000*abs(tolerance))
    bkg_min_boundary = selected_velocity - 2*tolerance
    vel_min_boundary = selected_velocity - tolerance
    vel_max_boundary = selected_velocity + tolerance
    bkg_max_boundary = selected_velocity + 2*tolerance

    # Compute SNR from vel_counts and bkg_counts
    counts, bins = np.histogram(sensor.df['velocity'], bins=nbins)
    bins = 0.5*(bins[:-1] + bins[1:])
    sig2bkg = counts[(bins >= bkg_min_boundary) & (bins < bkg_max_boundary)].sum()
    sig1bkg = counts[(bins >= vel_min_boundary) & (bins < vel_max_boundary)].sum()
    bkg = sig2bkg - sig1bkg
    signal = sig1bkg - bkg
    snr = signal/bkg
    
    fig = px.histogram(sensor.df['velocity'], log_y=True)
    fig.update_layout(
        xaxis_title="Radial Velocity [m/s]",
        yaxis_title="Counts",
        title_text="Velocity Histogram",
        font=dict(size=16),
    )
    fig.add_annotation(text=sample_text,
        xref="x domain",
        yref="y domain",
        x=0.01, y=1, showarrow=False,
        font=dict(color="red"),
        )
    value_str = "Signal = " + "%.0f"% signal + ", Bkg = " + "%.0f"% bkg
    fig.add_annotation(text=value_str,
        xref="x domain",
        yref="y domain",
        x=0.01, y=0.9, showarrow=False,
        font=dict(color="red"),
        )
    value_str = "Velocity_SNR = " + "%.1f"% snr
    fig.add_annotation(text=value_str,
        xref="x domain",
        yref="y domain",
        x=0.01, y=0.8, showarrow=False,
        font=dict(color="red"),
        )

    fig.add_vline(x=bkg_min_boundary, line_width=2, line_dash='dash', line_color='blue')
    fig.add_vline(x=vel_min_boundary, line_width=2, line_dash='dash', line_color='red')
    fig.add_vline(x=vel_max_boundary, line_width=2, line_dash='dash', line_color='red')
    fig.add_vline(x=bkg_max_boundary, line_width=2, line_dash='dash', line_color='blue')
    fig.write_html(f"0_{sensor.sensor_type}_{target_type}_{target_speed}_ctf_velocityhistogram.html")    
    fig.show()

def make_velocity_plots(df_all_frames, sensor, target_type, threshold_count=2):
    # Descr: Generate Velocity Analysis results plots

    # 'threshold_count' is used to remove frames which have 'count' below 'threshold_count', 
    # and for computing 'velocity_percent'.mean().

    sample_text = sensor.sensor_type + ', ' + target_type

    # Make plots which use points-data
    fig = px.scatter(df_all_frames, x='x', y='y', color='frame_idx')
    fig.update_layout(
        xaxis_title="Longitudinal Distance [m] (X-axis)",
        yaxis_title="Lateral Distance [m] (Y-axis)",
        title_text="Distance",
        font=dict(size=16),
    )
    fig.add_annotation(text=sample_text,
        xref="x domain",
        yref="y domain",
        x=0.01, y=1, showarrow=False,
        font=dict(color="red"),
        )
    fig.show()
    fig.write_html(f"1_{sensor.sensor_type}_{target_type}_ctf_distance_vs_frame.html")

    df_all_frames_stats = compute_statistics(df_all_frames, sensor)

    # Make plots which use modified stats-data
    # Remove frames which have 'counts' below threshold. Then plot and output 'velocity_percent' KPI's
    df_all_frames_stats_valid = df_all_frames_stats[df_all_frames_stats['count']>=threshold_count]
    df_all_frames_stats_valid['velocity_percent'] = df_all_frames_stats_valid['velocity_percent'].abs()

    fig = px.line(df_all_frames_stats_valid, x='range_mean', y='count', log_y=True)
    fig.update_layout(
        xaxis_title="Range_mean [m]",
        yaxis_title="Number of Points (filtered)",
        title_text="Number of Points",
        font=dict(size=16),
        )
    fig.add_annotation(text=sample_text,
        xref="x domain",
        yref="y domain",
        x=0.01, y=1, showarrow=False,
        font=dict(color="red"),
        )
    fig.add_hline(y=threshold_count, line_width=2, line_dash='dash', line_color='red')
    fig.show()
    fig.write_html(f"2_{sensor.sensor_type}_{target_type}_ctf_count_vs_range.html")

    fig = go.Figure(data=go.Scatter(
        x=df_all_frames_stats_valid['range_mean'],
        y=df_all_frames_stats_valid['velocity_mean'].abs(),
        error_y=dict(
            type='data',
            array=df_all_frames_stats_valid['velocity_std'],
            visible=True)
    ))
    fig.update_layout(
        xaxis_title="Range_mean [m]",
        yaxis_title="Radial Velocity_mean_abs [m/s]",
        title_text="Radial Velocity_mean",
        font=dict(size=16),
        )
    fig.add_annotation(text=sample_text,
        xref="x domain",
        yref="y domain",
        x=0.01, y=1, showarrow=False,
        font=dict(color="red"),
        )
    fig.show()
    fig.write_html(f"3_{sensor.sensor_type}_{target_type}_ctf_velocity_vs_range.html")
    # --------

    fig = px.line(df_all_frames_stats_valid, x='range_mean', y='velocity_percent')
    fig.update_layout(
        xaxis_title="Range_mean [m]",
        yaxis_title="Radial Velocity Noise_abs [%]",
        title_text="Radial Velocity Noise",
        font=dict(size=16),
    )
    fig.add_annotation(text=sample_text,
        xref="x domain",
        yref="y domain",
        x=0.01, y=1, showarrow=False,
        font=dict(color="red"),
        )
    value_str = "velocity_noise = " + "%.2f"% df_all_frames_stats_valid['velocity_percent'].mean() + "%"
    fig.add_annotation(text=value_str,
        xref="x domain",
        yref="y domain",
        x=0.01, y=0.9, showarrow=False,
        font=dict(color="red"),
        )
    fig.show()
    fig.write_html(f"4_{sensor.sensor_type}_{target_type}_ctf_velocitypercent_vs_range.html")

def make_hrl_velocity_plots(df_all_frames, df_all_frames_stats, sensor, target_type):
    if sensor.sensor_type == "ars542":
       pass

    elif sensor.sensor_type =="scantinel":
        pass

    elif sensor.sensor_type == "silc":
        pass

    elif sensor.sensor_type == "voyant":
        pass

    elif sensor.sensor_type == "HRL":
        sample_text = sensor.sensor_type + ', ' + target_type

        fig = px.scatter(df_all_frames, x='x', y='y', color='frame_idx')
        fig.update_layout(
            xaxis_title="Longitudinal Distance [m] (X-axis)",
            yaxis_title="Lateral Distance [m] (Y-axis)",
            title_text="Distance",
            font=dict(size=16),
        )
        fig.add_annotation(text=sample_text,
            xref="x domain",
            yref="y domain",
            x=0.01, y=1, showarrow=False,
            font=dict(color="red"),
            )
        fig.show()
        fig.write_html(f"1_{sensor.sensor_type}_{target_type}_ctf_distance_vs_frame.html")
    
        df_all_frames_stats.reset_index(inplace=True)
        df_all_frames_stats['velocity_abs']=df_all_frames_stats['velocity'].abs()

        # Flatten the multi-index columns
        df_all_frames_stats.columns = ['_'.join(col) for col in df_all_frames_stats.columns]

        fig = px.line(df_all_frames_stats, x='range_', y='x_count', log_y=True)
        fig.update_layout(
            xaxis_title="Range [m]",
            yaxis_title="Number of Points (filtered)",
            title_text="HRL Number of Points",
            font=dict(size=16),
            )
        fig.add_annotation(text=sample_text,
            xref="x domain",
            yref="y domain",
            x=0.01, y=1, showarrow=False,
            font=dict(color="red"),
            )
        fig.show()
        fig.write_html(f"2_{sensor.sensor_type}_{target_type}_ctf_count_vs_range.html")

        fig = px.line(df_all_frames_stats, x='range_', y='velocity_abs_', log_y=False)
        fig.update_layout(
            xaxis_title="Range [m]",
            yaxis_title="Radial Velocity_calculated_abs [m/s]",
            title_text="HRL Velocity",
            font=dict(size=16),
            )
        fig.add_annotation(text=sample_text,
            xref="x domain",
            yref="y domain",
            x=0.01, y=1, showarrow=False,
            font=dict(color="red"),
            )
        fig.show()
        fig.write_html(f"3_{sensor.sensor_type}_{target_type}_ctf_velocitycalc_vs_range.html")

def make_multisensor_velocity_plot(df_stats_ars, df_stats_hrl, df_stats_scantinel, df_stats_silc, target_type, target_speed):
    # Make plot of Multi-sensor data overlaid on one plot axes

    if target_speed == '2mph':
        vel_min = 0
        vel_max = 2
    elif target_speed == '22mph':
        vel_min = 8
        vel_max = 12
    elif target_speed == '65mph':
        vel_min = 28
        vel_max = 32
    elif target_speed == '19mph':
        vel_min = -11
        vel_max = -7

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_stats_ars['range_mean'],
        y=df_stats_ars['velocity_mean'].abs(),
        error_y=dict(
            type='data',
            array=df_stats_ars['velocity_std'],
            visible=True),
        mode='lines',
        name='ars_ctf'))
    fig.add_trace(go.Scatter(
        x=df_stats_scantinel['range_mean'],
        y=df_stats_scantinel['velocity_mean'].abs(),
        error_y=dict(
            type='data',
            array=df_stats_scantinel['velocity_std'],
            visible=True),
        mode='lines',
        name='scantinel_ctf'))
    fig.add_trace(go.Scatter(
        x=df_stats_silc['range_mean'],
        y=df_stats_silc['velocity_mean'].abs(),
        error_y=dict(
            type='data',
            array=df_stats_silc['velocity_std'],
            visible=True),
        mode='lines',
        name='silc_ctf'))
    fig.add_trace(go.Scatter(
        x=df_stats_hrl['range_'],
        y=df_stats_hrl['velocity_'].abs(),
        mode='lines',
        name='hrl_ctf'))
    fig.update_layout(
        xaxis_title="Range_mean [m]",
        yaxis_title="Radial Velocity_mean_abs [m/s]",
        title_text="Multi-sensor, Radial Velocity, " + target_type + ", " + target_speed,
        font=dict(size=16),
        )
    fig.update_layout(yaxis_range=[vel_min, vel_max])
    fig.show()
    fig.write_html(f"Multi-sensor_{target_type}_{target_speed}_ctf_velocity_mean_abs_vs_range.html")

def make_hrl_velocity_difference_plot(df_stats_hrl, df_stats_ars, target_speed):
    # This block is for computing and plotting Velocity-Difference
    # It requires HRL data and ARS data as input.

    # from plotly.subplots import make_subplots

    if target_speed == '2mph':
        vel_min = 0.2
        vel_max = 1.8
    elif target_speed == '22mph':
        vel_min = 7
        vel_max = 11
    elif target_speed == '65mph':
        vel_min = 20
        vel_max = 32
    elif target_speed == '19mph':
        vel_min = -10
        vel_max = 0

    lookuptable_ars = df_stats_ars[['range_mean', 'velocity_mean']].copy()
    lookuptable_hrl = df_stats_hrl[['range_', 'velocity_']].copy()
    lookuptable_hrl['velocity_diff'] = np.nan   # Add a new column 'velocity_diff'
    lookuptable_sorted_ars = lookuptable_ars.sort_values(by=['range_mean']) # sort by 'range_mean'

    # Use 'lookuptable_ars' to compute 'v_delta' values corresponding to HRL 'range_' values.
    v_delta = pd.Series()
    for idx, rr in enumerate(lookuptable_hrl['range_']):
        new_val = lookuptable_sorted_ars.iloc[(lookuptable_sorted_ars['range_mean']-rr).abs().argsort()[:1]]['velocity_mean']
        new_val = new_val.tolist()[0] - lookuptable_hrl['velocity_'][idx]
        v_delta[len(v_delta)] = new_val

    idx_start = 30   # Discard the first few HRL points which are determined to be 'bad' by inspection
    idx_end = -5    # Discard the last few HRL points which are determined to be 'bad' by inspection
    lookuptable_hrl = lookuptable_hrl[idx_start:idx_end]
    lookuptable_hrl['velocity_diff'] = v_delta[idx_start:idx_end]

    # Plot the HRL data (lookuptable_hrl), including 'v_delta'
    fig = sp.make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=lookuptable_hrl['range_'],
        y=lookuptable_hrl['velocity_'],
        name="Velocity"),
        secondary_y=False,
    )
    fig.update_traces(line_color='#AB63FA')
    fig.add_trace(go.Scatter(
        x=lookuptable_hrl['range_'],
        y=lookuptable_hrl['velocity_diff'],
        name="Velocity_diff"),
        secondary_y=True,
    )
    # Add figure title
    fig.update_layout(
        title_text="HRL Velocity and Velocity_difference"
    )
    # Set x-axis title
    fig.update_xaxes(title_text="<b>Range [m]</b>")
    # Set y-axes titles
    fig.update_yaxes(
        title_text="<b>Velocity [m/s]</b>", 
        title_font_color='#AB63FA',
        range=[vel_min, vel_max],      # this y-axis range is hard-coded for 22 mph
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="<b>Velocity_difference [m/s]</b>", 
        title_font_color="red",
        secondary_y=True,
    )
    value_str = "mean = " + "%.3f"% v_delta[idx_start:].mean() + " m/s"
    fig.add_annotation(text=value_str,
        xref="x domain",
        yref="y domain",
        x=0.01, y=0.95, showarrow=False,
        font=dict(color="red"),
        )
    value_str = "std = " + "%.3f"% v_delta[idx_start:].std() + " m/s"
    fig.add_annotation(text=value_str,
        xref="x domain",
        yref="y domain",
        x=0.01, y=0.9, showarrow=False,
        font=dict(color="red"),
        )

    fig.show()
    fig.write_html(f"5_HRL_10P_{target_speed}_F_ctf_velocity_diff_vs_range.html")