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
import csv
import platform

if platform.system() != "Darwin":
    from ecal.measurement.measurement import Measurement
    from selda.pc2_processor import PC2Processor
    import selda.pointfield_types_dicts as pointfield_dicts


import warnings
warnings.filterwarnings('ignore')

logging.basicConfig()
logger = logging.getLogger(' ')
logger.setLevel(logging.INFO)

session_info.show()

def csv_to_dict():
    # hessai_dict = {}
    # with open('Hessai_channels.csv', 'r') as csv_file:
    #     csv_reader = csv.DictReader(csv_file)
    #     for row in csv_reader:
    #         key = row.pop('channel')
    #         hessai_dict[key] = row

    hessai_dict = pd.read_csv('Hessai_channels.csv')
    return hessai_dict


# %%
def get_current_time_date():
    curr = time.time()
    obj = time.localtime(curr)
    time_date = str(obj[3])+"."+str(obj[4])+"."+str(obj[5])+"_"+str(obj[1])+ "."+str(obj[2])+"."+str(obj[0])
    return time_date

# %%


# %%
def calculate_azimuth_elevation(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    elevation = np.arcsin(z / r)
    azimuth = -np.arctan2(y, x)
    return np.degrees(azimuth), np.degrees(elevation)

# %%
class sensor:
    def __init__(self, sensor_type, directory, debug=False, target_type=None):
        # Basic variables ------
        self.sensor_type = sensor_type
        directory_aux = os.path.normpath(directory)
        directory_aux = os.path.join(directory_aux, "")
        self.directory = directory_aux
        self.debug = debug
        path = self.directory.split(os.sep)
        try:
            self.data_name = path[-3] + "_" + path[-2]
        except:
            self.data_name = path[-2]
        self.pkl_file_directory = self.directory + "All_frames.pkl"
        self.df_single_frame = pd.DataFrame()
        self.shots_in_frame = 0
        self.df = pd.DataFrame()
        self.df_selection  = pd.DataFrame()
        # Extended variables ------
        # tecnically not needed but can be used for debbugging specially on early development. (in order of process)
        if debug:
            self.df_csv_append = pd.DataFrame() # Contains all the data read from the input file ("raw")
            self.df_all_frames = pd.DataFrame() # Contains all returns after modifying the data ("processed part 1")
            self.df_valid_returns = pd.DataFrame() # Contains all the valid returns from the 100 frames ("processed part 2")
            self.df_all_frames_target = pd.DataFrame() # contains all the data after selecting a target
        # "self.df" should contain the current data information (as a df). so it should match the variables shown above at different parts of the program.
        # Sensor information ------
        if self.sensor_type == "voyant":
            self.hres = 0.04
            self.vres = 0.97
            self.hfov = 30
            self.vfov = 30
        elif self.sensor_type == "scantinel":
            self.hfov = 20
            self.hres = 0.2
            self.vfov = 26
            self.vres = 0.1
        elif self.sensor_type == "silc":
            self.hres = 0.01
            self.vres = 0.084
            self.hfov = 20
            self.vfov = 8.6
        elif self.sensor_type == "HRL":
            # High speed resolution
            self.hres = 0.195
            self.vres = 0.1
            self.hres_per = 0.391
            self.vres_per = 0.1
            self.hfov = 20
            self.vfov = 8.6
        elif self.sensor_type == "ars542":
            # hres = 0.04
            # vres = 0.97
            # hfov = 30
            # vfov = 30
            pass
        elif self.sensor_type == "Hessai":
            self.hres_A=0.8
            self.hres_B=0.4
            self.hfov=360
            self.vfov=105.2

    def export_point_data(self, folder_name="test"):
        try:
            # Path 
            folder_path = os.path.join(self.directory, folder_name) 
            os.mkdir(folder_path)
        except:
            logger.warning("Error making folder and exporting data!")
        self.df.to_csv(f"{folder_path}/{get_current_time_date()}_{self.data_name}_data.txt", sep=',')

    def data_read(self, multiple_files= False):
        df_csv_append = pd.DataFrame()

        if self.sensor_type == "voyant":
            csv_files = glob.glob('{}*.csv'.format(self.directory))
            
            # append the CSV files
            nn=1
            for file in csv_files:
                print(f'{nn}: {file}')
                nn += 1
            print('Enter the number of the file to read.\n')
            nn = input()
            filename = csv_files[int(nn)-1]
            print(f"Reading: {filename}")

            df = pd.read_csv(file)
            df_csv_append = pd.concat([df_csv_append, df], ignore_index=True)
            # break # so we only read 1 csv, since all frames are already joined.

        elif self.sensor_type == "scantinel":
            hdf5_files = glob.glob('{}*.hdf5'.format(self.directory))

            # Select first hdf5 file
            meas = Measurement(hdf5_files[0])
            proc = PC2Processor(verbose=False)
            
            df_csv_append = pd.DataFrame()

            frame_counter = 0
            # iterate through each message/frame
            for ts, message in meas["clara_pcl_corrected"]:          
                # only run on 100 frames
                if frame_counter >= 100:
                    break

                proc.process_pointcloud2(message, pad_dtypes=True)

                # convert to dataframe
                df_single_frame = proc.to_df()

                # attach frame id column
                # df_single_frame["frame_idx"] = message.header.frame_id
                df_single_frame["frame_idx"] = frame_counter


                # append frame to df
                df_csv_append = pd.concat([df_csv_append, df_single_frame])

                frame_counter+= 1


        elif self.sensor_type == "silc":
            # frame_id = 0
            # csv_files = glob.glob('{}*.pcd'.format(self.directory))
            # df_data = pd.DataFrame()
            # df_header = pd.DataFrame()
            # for file in csv_files:
            #     # print(file)
            #     df_header = pd.read_csv(file, skiprows=2, nrows=1, header=None, sep=' ').drop(0, axis=1)
            #     df_header.columns = [0,1, 2, 3, 4, 5, 6, 7, 8]
            #     df_header['frame'] = pd.Series([frame_id for x in range(len(df_header.index))])
            #     exit
                
            # for file in csv_files:
            #     df_data = pd.read_csv(file, skiprows=11, header=None, sep=' ')
            #     #drop nans here to make it faster
            #     df_data['frame'] = pd.Series([frame_id for x in range(len(df_data.index))])
            #     df_csv_append = pd.concat([df_csv_append, df_data] , ignore_index=True)
            #     # df_csv_append = df_csv_append.append(df_data, ignore_index=True)
            #     frame_id = frame_id + 1
            
            # # df_csv_append = df_header.append(df_csv_append, ignore_index=True)
            # df_csv_append = pd.concat([df_header, df_csv_append] , ignore_index=True)
            # df_csv_append.columns = df_csv_append.iloc[0]
            # df_csv_append.drop(df_csv_append.index[0], axis=0, inplace=True)
            # df_csv_append.rename(columns={0: 'frame_idx'}, inplace=True)

            hdf5_files = glob.glob('{}*.hdf5'.format(self.directory))
            df_csv_append = pd.DataFrame()

            if multiple_files == False:
                meas = Measurement(hdf5_files[0])
                proc = PC2Processor(verbose=False)

                frame_counter = 0
                # iterate through each message/frame
                for ts, message in meas["SiLCPointCloudPb"]:          
                    # only run on 100 frames
                    if frame_counter >= 100:
                        break

                    proc.process_pointcloud2(message, pad_dtypes=True)

                    # convert to dataframe
                    df_single_frame = proc.to_df()

                    # attach frame id column
                    # df_single_frame["frame_idx"] = message.header.frame_id
                    df_single_frame["frame_idx"] = frame_counter


                    # append frame to df
                    df_csv_append = pd.concat([df_csv_append, df_single_frame])

                    frame_counter+= 1

            elif multiple_files == True:
                frame_counter = 0
                file_count = 0
                for file in hdf5_files:
                    logger.info(f'File started ({file_count+1}/{len(hdf5_files)}): {file}') 
                    meas = Measurement(file)
                    proc = PC2Processor(verbose=False)

                    # iterate through each message/frame
                    for ts, message in meas["SiLCPointCloudPb"]:    
                        # print(f'frame idx = {frame_counter}')      
                        # only run on 100 frames
                        # if frame_counter >= 100:
                        #     break

                        proc.process_pointcloud2(message, pad_dtypes=True)

                        # convert to dataframe
                        df_single_frame = proc.to_df()

                        # attach frame id column
                        # df_single_frame["frame_idx"] = message.header.frame_id
                        df_single_frame["frame_idx"] = frame_counter


                        # append frame to df
                        df_csv_append = pd.concat([df_csv_append, df_single_frame])

                        frame_counter+= 1
                    file_count = file_count + 1


        elif self.sensor_type == "HRL":
            hdf5_files = glob.glob('{}*.hdf5'.format(self.directory))

            # Select first hdf5 file
            meas = Measurement(hdf5_files[0])
            proc = PC2Processor(verbose=False)
            
            df_csv_append = pd.DataFrame()

            frame_counter = 0
            # iterate through each message/frame
            for ts, message in meas["AEyeSensorPointCloudData"]:
                if frame_counter == 0: 
                    pass
                    # print(type(message))
                frame_counter+= 1
                
                # only run on 100 frames
                if frame_counter > 100:
                    break

                # convert to pandas array
                proc.process_pointcloud2(message, pad_dtypes=True)

                # convert to dataframe
                df_single_frame = proc.to_df()

                #print(df_single_frame)

                # attach frame id column
                df_single_frame["frame_idx"] = message.header.frame_id

                # append frame to df
                df_csv_append = pd.concat([df_csv_append, df_single_frame])

        elif self.sensor_type == "ars542":
            hdf5_files = glob.glob('{}*.hdf5'.format(self.directory))

            # Select first hdf5 file
            meas = Measurement(hdf5_files[0])
            proc = PC2Processor(verbose=False)
            
            df_csv_append = pd.DataFrame()

            frame_counter = 0
            # iterate through each message/frame
            for ts, message in meas["ARS540PointCloudPb"]:
                if frame_counter == 0: 
                    pass

                frame_counter+= 1
                
                # only run on 100 frames
                if frame_counter > 300:
                    break

                # convert to pandas array
                proc.process_pointcloud2(message, pad_dtypes=True)

                # convert to dataframe
                df_single_frame = proc.to_df()

                # attach frame id column
                df_single_frame["frame_idx"] = frame_counter

                # append frame to df
                df_csv_append = pd.concat([df_csv_append, df_single_frame])

        elif self.sensor_type == "SRL":
            hdf5_files = glob.glob('{}*.hdf5'.format(self.directory))

            # Select first hdf5 file
            meas = Measurement(hdf5_files[0])
            proc = PC2Processor(verbose=False)
            
            df_csv_append = pd.DataFrame()

            frame_counter = 0
            # iterate through each message/frame
            for ts, message in meas["VPR002_PointCloudPb"]:
                if frame_counter == 0: 
                    pass
                    # print(type(message))
                frame_counter+= 1            # only run on 100 frames
                if frame_counter > 100:
                    break

                # ############ MARTINS HACKK
                # message.fields.add()
                # message.fields[-1].datatype = pointfield_dicts.DATATYPE_STR_TO_INT_DICT["uint16"]
                # message.fields[-1].name = "histogram"
                # # Calculate the offset dynamically from the last field
                # message.fields[-1].offset = message.fields[-2].offset + pointfield_dicts.DATATYPES_DICT_BYTES_SIZE[message.fields[-2].datatype]
                # message.fields[-1].count = 400
                # message.fields[-2].datatype = 4
                message.fields[-1].offset = 7
                # #############################

                # convert to pandas array
                proc.process_pointcloud2(message, pad_dtypes=True)

                # convert to dataframe
                df_single_frame = proc.to_df()

                #print(df_single_frame)

                # attach frame id column
                df_single_frame["frame_idx"] = frame_counter-1

                # append frame to df
                df_csv_append = pd.concat([df_csv_append, df_single_frame])
        

        elif self.sensor_type == "Hessai":
            hdf5_files = glob.glob('{}*.hdf5'.format(self.directory))

            # Select first hdf5 file
            meas = Measurement(hdf5_files[0])
            proc = PC2Processor(verbose=False)
            
            df_csv_append = pd.DataFrame()

            frame_counter = 0
            # iterate through each message/frame
            for ts, message in meas["QT128PointCloudPb"]:
                if frame_counter == 0: 
                    pass
                    # print(type(message))
                frame_counter+= 1
                # only run on 100 frames
                if frame_counter > 10:
                    break

                # convert to pandas array
                proc.process_pointcloud2(message, pad_dtypes=True)

                # convert to dataframe
                df_single_frame = proc.to_df()

                #print(df_single_frame)

                # attach frame id column
                df_single_frame["frame_idx"] = frame_counter-1

                # append frame to df
                df_csv_append = pd.concat([df_csv_append, df_single_frame])

        elif self.sensor_type == "opsys":
            hdf5_files = glob.glob('{}*.hdf5'.format(self.directory))

            # Select first hdf5 file
            meas = Measurement(hdf5_files[0])
            proc = PC2Processor(verbose=False)
            
            df_csv_append_188 = pd.DataFrame()
            df_csv_append_189 = pd.DataFrame()
            df_csv_append_190 = pd.DataFrame()
            df_csv_append_191 = pd.DataFrame()

            frame_counter = 0
            # iterate through each message/frame
            for ts, message in meas["Opsys_188_PointCloudPb"]:
                if frame_counter == 0: 
                    pass
                frame_counter+= 1
                # only run on 100 frames
                if frame_counter > 300:
                    break
                # convert to pandas array
                proc.process_pointcloud2(message, pad_dtypes=True)
                # convert to dataframe
                df_single_frame = proc.to_df()
                # attach frame id column
                df_single_frame["frame_idx"] = frame_counter
                # append frame to df
                df_csv_append_188 = pd.concat([df_csv_append_188, df_single_frame])

            frame_counter = 0
            # iterate through each message/frame
            for ts, message in meas["Opsys_189_PointCloudPb"]:
                if frame_counter == 0: 
                    pass
                frame_counter+= 1
                # only run on 100 frames
                if frame_counter > 300:
                    break
                # convert to pandas array
                proc.process_pointcloud2(message, pad_dtypes=True)
                # convert to dataframe
                df_single_frame = proc.to_df()
                # attach frame id column
                df_single_frame["frame_idx"] = frame_counter
                # append frame to df
                df_csv_append_189 = pd.concat([df_csv_append_189, df_single_frame])

            frame_counter = 0
            # iterate through each message/frame
            for ts, message in meas["Opsys_190_PointCloudPb"]:
                if frame_counter == 0: 
                    pass
                frame_counter+= 1
                # only run on 100 frames
                if frame_counter > 300:
                    break
                # convert to pandas array
                proc.process_pointcloud2(message, pad_dtypes=True)
                # convert to dataframe
                df_single_frame = proc.to_df()
                # attach frame id column
                df_single_frame["frame_idx"] = frame_counter
                # append frame to df
                df_csv_append_190 = pd.concat([df_csv_append_190, df_single_frame])

            frame_counter = 0
            # iterate through each message/frame
            for ts, message in meas["Opsys_191_PointCloudPb"]:
                if frame_counter == 0: 
                    pass
                frame_counter+= 1
                # only run on 100 frames
                if frame_counter > 300:
                    break
                # convert to pandas array
                proc.process_pointcloud2(message, pad_dtypes=True)
                # convert to dataframe
                df_single_frame = proc.to_df()
                # attach frame id column
                df_single_frame["frame_idx"] = frame_counter
                # append frame to df
                df_csv_append_191 = pd.concat([df_csv_append_191, df_single_frame])

            df_all = [df_csv_append_188, df_csv_append_189, df_csv_append_190, df_csv_append_191]
            df_csv_append = pd.concat(df_all)

        # elif self.sensor_type == "NA":
            # pass

        df_csv_append.to_pickle(self.pkl_file_directory)

        self.df = df_csv_append
        if self.debug:
            self.df_csv_append = df_csv_append # repeated just for debugging
        del df_csv_append

        logger.info(f' data read complete!')
        # return df_csv_append


    def row_col_fix(self):
        # Inputs ------   
        df_all_frames = self.df # or self.df_csv_append
        #  ------
        # This function expects 'x', 'y', and 'z' to be present for all sensor_types.
        # It computes azimuth, elevation, rol and col. It also ensures that all sensor_types
        # have 'range' and 'velocity' columns whenever possible.
        # NOTE: SRL is different from other sensor_types because it has a column for 'range_calculated'.

        df_support = pd.DataFrame()

        # Add column for 'range' if not already present. This applies to all sensor_types.
        if 'range' not in df_all_frames:
            if ('x' in df_all_frames) & ('y' in df_all_frames) & ('z' in df_all_frames):
                df_all_frames['range'] = (df_all_frames['x']**2 + df_all_frames['y']**2 + df_all_frames['z']**2)**.5

        if self.sensor_type == "voyant":
            # We are using the "point_idx" value and the range to determine same Tx directions
            df_support  = df_all_frames.groupby(['frame_idx']).size().reset_index().rename(columns={0:'count'})
            # Add 'velocity' column if not already present
            if ('velocity' not in df_all_frames) & ('radial_vel' in df_all_frames):
                df_all_frames['velocity'] = - df_all_frames['radial_vel']
            df_all_frames['azimuth_angle'] = np.degrees(np.arctan2(df_all_frames['y'], df_all_frames['x']))
            df_all_frames['elevation_angle'] = np.degrees(np.arcsin(df_all_frames['z'] / np.sqrt(df_all_frames['x']**2 + df_all_frames['y']**2 + df_all_frames['z']**2)))

        elif self.sensor_type == "scantinel":
            # Column and row already exist in data.
            df_support  = df_all_frames.groupby(['frame_idx']).size().reset_index().rename(columns={0:'count'})
            # Add 'velocity' column if not already present
            if ('velocity' not in df_all_frames) & ('Velocity' in df_all_frames):
                df_all_frames['velocity'] = df_all_frames['Velocity']
            # Add 'range' column if not already present
            if ('Range' in df_all_frames):
                df_all_frames['range'] = df_all_frames['Range']
            
            def modify_azimuth(row):
                first_frame_lean = "Right"
                if row['frame_idx'] % 2 == 0:
                    if first_frame_lean == 'Right':
                        return row['Azimuth'] - ((row['Elevation'] / 0.1) * 0.0135)
                    else:
                        return row['Azimuth'] + ((row['Elevation'] / 0.1) * 0.0135)
                else:
                    if first_frame_lean == 'Right':
                        return row['Azimuth'] + ((row['Elevation'] / 0.1) * 0.0135)
                    else:
                        return row['Azimuth'] - ((row['Elevation'] / 0.1) * 0.0135)

            df_all_frames['azimuth_modified'] = df_all_frames.apply(modify_azimuth, axis=1)

            self.hfov = 20
            self.hres = 0.2
            self.vfov = 26
            self.vres = 0.1
            df_all_frames['col'] = round(((df_all_frames['azimuth_modified'] + (self.hfov/2)) * (self.hfov/self.hres)) / self.hfov)
            df_all_frames['row'] = round(((df_all_frames['Elevation'] + (self.vfov/2)) * (self.vfov/self.vres)) / self.vfov)

            # Add 'azimuth_angle' column if not already present
            if ('azimuth_angle' not in df_all_frames) & ('azimuth_modified' in df_all_frames):
                df_all_frames['azimuth_angle'] = df_all_frames['azimuth_modified']
            # Add 'elevation_angle' column if not already present
            if ('elevation_angle' not in df_all_frames) & ('Elevation' in df_all_frames):
                df_all_frames['elevation_angle'] = df_all_frames['Elevation']

        elif self.sensor_type == "silc": # this is new
            self.hres = 0.01
            self.vres = 0.084
            self.hfov = 20
            self.vfov = 8.6
            # Adding col-row
            cast_tosensor_type = {
            'azimuth_angle': float,
            'elevation_angle': float
            }
            df_all_frames = df_all_frames.astype(cast_tosensor_type)
            elevation_offset = 7.1
            df_all_frames['col'] = round(((df_all_frames['azimuth_angle'] + (self.hfov/2)) * (self.hfov/self.hres)) / self.hfov)
            df_all_frames['row'] = round(((df_all_frames['elevation_angle'] + (elevation_offset)) * (self.vfov/self.vres)) / self.vfov)
            # fixing coordinate system
            df_all_frames.rename(columns={'x': 'y_2'}, inplace=True)
            df_all_frames.rename(columns={'y': 'x'}, inplace=True)
            df_all_frames.rename(columns={'y_2': 'y'}, inplace=True)
            df_all_frames['y'] = - df_all_frames['y']
            df_support  = df_all_frames.groupby(['frame_idx']).size().reset_index().rename(columns={0:'count'})
            # Add column for 'range'
            df_all_frames['range'] = (df_all_frames['x']**2 + df_all_frames['y']**2 + df_all_frames['z']**2)**.5

        elif self.sensor_type == "HRL":
            # Column and row already exist in data.
            df_support  = df_all_frames.groupby(['frame_idx']).size().reset_index().rename(columns={0:'count'})
            # pass

        elif self.sensor_type == "ars542":
            df_support  = df_all_frames.groupby(['frame_idx']).size().reset_index().rename(columns={0:'count'})
            # Add 'velocity' column if not already present
            if ('velocity' not in df_all_frames) & ('f_vrelRad' in df_all_frames):
                df_all_frames['velocity'] = df_all_frames['f_vrelRad']
            # Add 'range' column if not already present
            if ('range' not in df_all_frames) & ('f_rangeRad' in df_all_frames):
                df_all_frames['range'] = df_all_frames['f_rangeRad']
            # Add 'azimuth_angle' column if not already present
            if ('azimuth_angle' not in df_all_frames) & ('f_azAng' in df_all_frames):
                df_all_frames['azimuth_angle'] = df_all_frames['f_azAng']
            # Add 'elevation_angle' column if not already present
            if ('elevation_angle' not in df_all_frames) & ('f_elevAng' in df_all_frames):
                df_all_frames['elevation_angle'] = df_all_frames['f_elevAng']
        
        elif self.sensor_type == "SRL":
            # Column and row already exist in data.
            df_support  = df_all_frames.groupby(['frame_idx']).size().reset_index().rename(columns={0:'count'})
            # pass

        elif self.sensor_type == "Hessai":
            hres_A=0.8
            hres_B=0.4
            vres=0.14
            hfov=360
            vfov=105.2

            hessai_dict = csv_to_dict()
            hessai_df=pd.DataFrame(hessai_dict)
            # CALCULATION OF AZIMUTH AND ELEVATION 
            df_all_frames['azimuth_angle'], df_all_frames['elevation_angle'] = \
            zip(*df_all_frames.apply(lambda row: calculate_azimuth_elevation(row['x'], row['y'], row['z']), axis=1))
            
            #COPY SECTIONS FROM A DF TO ANTOHER
            #  df_support_hfov_A = df_all_frames[(df_all_frames['ring'] <= 64)].copy()
            #  df_support_hfov_B = df_all_frames[(df_all_frames['ring'] > 64)].copy()
            df_all_frames['col'] = round(((df_all_frames['azimuth_angle'] + (hfov/2)) * (hfov/hres_A)) / hfov)
        
            df_all_frames['row'] = round(((df_all_frames['elevation_angle'] + (vfov/2)) * (vfov/vres)) / vfov)
            df_support  = df_all_frames.groupby(['frame_idx']).size().reset_index().rename(columns={0:'count'})

        # elif self.sensor_type == "Hessai":
        #     hres_A=0.8
        #     hres_B=0.4
        #     vres=0.14
        #     hfov=360
        #     vfov=105.2

        #     hessai_dict = csv_to_dict()
        #     hessai_df=pd.DataFrame(hessai_dict)
        #     # CALCULATION OF AZIMUTH AND ELEVATION 
        #     df_all_frames['azimuth_angle'], df_all_frames['elevation_angle'] = \
        #     zip(*df_all_frames.apply(lambda row: calculate_azimuth_elevation(row['x'], row['y'], row['z']), axis=1))
            
        #     #COPY SECTIONS FROM A DF TO ANTOHER
        #     #  df_support_hfov_A = df_all_frames[(df_all_frames['ring'] <= 64)].copy()
        #     #  df_support_hfov_B = df_all_frames[(df_all_frames['ring'] > 64)].copy()
        #     df_all_frames['col'] = round(((df_all_frames['azimuth_angle'] + (hfov/2)) * (hfov/hres_A)) / hfov)
        
        elif self.sensor_type == "opsys":
            # Column and row already exist in data.
            df_support  = df_all_frames.groupby(['frame_idx']).size().reset_index().rename(columns={0:'count'})
            # pass
        
    
        
            
        with open(self.pkl_file_directory, "rb") as pkl_file:
                with bz2.BZ2File(f"{self.directory}/all_data_compressed.pkl.bz2", "wb", compresslevel=9) as bz2_file:
                    bz2_file.write(pkl_file.read())

        pkl_file.close()
        bz2_file.close()

        # df_valid_returns = pd.read_pickle(f"{sensor.directory}/{pickle_filename}.pkl")
        # else:
        #      df_valid_returns = sensor.df
        
    # ------

        # Outputs ------
        self.shots_in_frame = df_support['count'].max()
        self.df = df_all_frames
        if self.debug:
            self.df_all_frames = df_all_frames # repeated just for debugging
        #  ------
        del df_all_frames, df_support
        logger.info(f' row_col_fix complete!')
        # return df_all_frames, shots_per_frame

    def data_validation(self, frame_number = 5, compressed=False):
        # Inputs ------
        if compressed == True:
            pickle_filename = "all_data_uncompressed"
            with bz2.BZ2File(f"{self.directory}/all_data_compressed.pkl.bz2", "rb") as bz2_file:
                with open(f"{self.directory}/{pickle_filename}.pkl", "wb") as pkl_file:
                    pkl_file.write(bz2_file.read())
        
            # Close the pkl file object and the BZ2File object.
            pkl_file.close()
            bz2_file.close()
            df_valid_returns = pd.read_pickle(f"{self.directory}/{pickle_filename}.pkl")
        else:
            df_valid_returns = self.df
            
        # ------

        if self.sensor_type == "voyant":
            df_valid_returns = df_valid_returns[(df_valid_returns['x'] != 0) & (df_valid_returns['y'] != 0) & (df_valid_returns['z'] != 0)]
            # Obtain the amount of shots included in the first frame of the recording.
            number_of_shots_in_frame = df_valid_returns.frame_idx.eq(df_valid_returns['frame_idx'].iloc[frame_number]).sum() #might not be needed
            frame_number_one = df_valid_returns[df_valid_returns['frame_idx'] == frame_number].copy()
            # "frame_number_one" contains only the frist frame.
            # frame_number_one = df_valid_returns.iloc[:number_of_shots_in_frame]
            
        elif self.sensor_type == "scantinel":
            df_valid_returns = df_valid_returns[(df_valid_returns['x'] != 0) & (df_valid_returns['y'] != 0) & (df_valid_returns['z'] != 0)]
            df_valid_returns.rename(columns={' Frame ID': 'Frame_ID'}, inplace=True)
            number_of_shots_in_frame = df_valid_returns.frame_idx.eq(df_valid_returns['frame_idx'].iloc[frame_number]).sum() #might not be needed
            frame_number_one = df_valid_returns[df_valid_returns['frame_idx'] == frame_number].copy()
            # pass

        elif self.sensor_type == "silc":
            df_valid_returns = df_valid_returns.replace(np.nan, None)
            df_valid_returns.dropna(subset=['x', 'y', 'z'], inplace=True)
            df_valid_returns = df_valid_returns[(df_valid_returns['x'] != 0) & (df_valid_returns['y'] != 0) & (df_valid_returns['z'] != 0)]
            # Obtain the amount of shots included in the first frame of the recording.
            number_of_shots_in_frame = df_valid_returns.frame_idx.eq(df_valid_returns['frame_idx'].iloc[0]).sum()
            frame_number_one = df_valid_returns[df_valid_returns['frame_idx'] == frame_number].copy()

        elif self.sensor_type == "HRL":
            df_valid_returns = df_valid_returns[(df_valid_returns['x'] != 0) & (df_valid_returns['y'] != 0) & (df_valid_returns['z'] != 0)]
            df_valid_returns.rename(columns={' frame_idx': 'frame_idx'}, inplace=True)
            number_of_shots_in_frame = df_valid_returns.frame_idx.eq(df_valid_returns['frame_idx'].iloc[frame_number]).sum() 
            # TODO: need to fix frame number
            frame_number_one = df_valid_returns[df_valid_returns['frame_idx'] == str(int(df_valid_returns['frame_idx'].min())+frame_number)].copy()

        elif self.sensor_type == "ars542":
            df_valid_returns = df_valid_returns[(df_valid_returns['x'] != 0) & (df_valid_returns['y'] != 0) & (df_valid_returns['z'] != 0)]
            # Obtain the amount of shots included in the first frame of the recording.
            number_of_shots_in_frame = df_valid_returns.frame_idx.eq(df_valid_returns['frame_idx'].iloc[frame_number]).sum() #might not be needed
            frame_number_one = df_valid_returns[df_valid_returns['frame_idx'] == frame_number].copy()

        elif self.sensor_type == "SRL":
            #df_valid_returns = df_valid_returns[(df_valid_returns['x'] != 0) & (df_valid_returns['y'] != 0) & (df_valid_returns['z'] != 0)]
            df_valid_returns.rename(columns={' frame_idx': 'frame_idx'}, inplace=True)
            number_of_shots_in_frame = df_valid_returns.frame_idx.eq(df_valid_returns['frame_idx'].iloc[frame_number]).sum() 
            ## FOR DETECTING MAX DISTANCE STOP SIGN, HIGH INTENSITY POINTS WONT SHOW AT LARGE DISTANCES, IN GENERAL ONLY STOP SIGN POINTS WILL APPEAR
            #frame_number_one=df_valid_returns[(df_valid_returns['frame_idx']==5) & (df_valid_returns['intensity']>75)].copy()
            ##Normal sampling of a given frame number
            frame_number_one=df_valid_returns[(df_valid_returns['frame_idx']==15)].copy()
            df_valid_returns['range_calculated'] = (df_valid_returns['x'] ** 2 + df_valid_returns['y'] ** 2 + df_valid_returns['z'] ** 2) ** 0.5
            # df_valid_returns['range'] = (df_valid_returns['x'])

        elif self.sensor_type == "Hessai":
            df_valid_returns = df_valid_returns[(df_valid_returns['x'] != 0) & (df_valid_returns['y'] != 0) & (df_valid_returns['z'] != 0)]
            df_valid_returns.rename(columns={' frame_idx': 'frame_idx'}, inplace=True)
            number_of_shots_in_frame = df_valid_returns.frame_idx.eq(df_valid_returns['frame_idx'].iloc[frame_number]).sum() 
            # TODO: need to fix frame number
            frame_number_one=df_valid_returns[(df_valid_returns['frame_idx']==5)].copy()
            # frame_number_one = df_valid_returns[df_valid_returns['frame_idx'] == str(int(df_valid_returns['frame_idx'].min())+frame_number)].copy()

        elif self.sensor_type == "opsys":
            df_valid_returns = df_valid_returns[(df_valid_returns['x'] != 0) & (df_valid_returns['y'] != 0) & (df_valid_returns['z'] != 0)]
            # Obtain the amount of shots included in the first frame of the recording.
            number_of_shots_in_frame = df_valid_returns.frame_idx.eq(df_valid_returns['frame_idx'].iloc[frame_number]).sum() #might not be needed
            frame_number_one = df_valid_returns[df_valid_returns['frame_idx'] == frame_number].copy()

        # Outputs ------
        self.df_single_frame = frame_number_one
        self.df = df_valid_returns
        # self.df = self.df_valid_returns
        if self.debug:
            self.df_valid_returns = df_valid_returns # repeated just for debugging
        # ------
        del frame_number_one, df_valid_returns
        logger.info(f' Data validation complete!')
        # return frame_number_one, self.df_valid_returns

#########################################################################################

# %%


# %%


# %%

