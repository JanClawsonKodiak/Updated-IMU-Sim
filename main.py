"""
Author: Jan Clawson
This is my implementation of a library made by @author: dongxiaoguang at Aceinna

GNSS-INS-SIM is an GNSS/INS simulation project, which generates reference trajectories, IMU sensor output, GPS output,
odometer output and magnetometer output. Users choose/set up the sensor model, define the waypoints and provide
algorithms, and gnss-ins-sim can generate required data for the algorithms, run the algorithms, plot simulation results,
save simulations results, and generate a brief summary.
"""

import os
import math
import numpy as np
from gnss_ins_sim.sim import imu_model
from gnss_ins_sim.sim import ins_sim
from demo_algorithms import free_integration_odo
from demo_algorithms import free_integration
from time import time

# globals
D2R = math.pi / 180  # conversion from degree to radian
fs = 100  # IMU sample frequency


def imu_creation(arw_input=.25, odo_scale=.999, odo_noise=.1, bias_input=3.5):
    """
    Create the IMU object from the library. Currently setup to operate as a 6 axis IMU, allowing odometer input, without GPS.
    There are three built-in IMU models: 'low-accuracy', 'mid-accuracy' and 'high accuracy'.

    Inputs: Currently, the gyro angle random walk, odometer scale error, odometer noise, and gyro bias instability. These
    could be expanded to include other sources of error, for instance in the accelerometer.

    Returns: An IMU object

    """
    # IMU model, typical for IMU381ZA
    imu_err = {
        'gyro_b': np.array([0.0, 0.0, 0.0]),
        # gyro angle random walk, deg/rt-hr
        'gyro_arw': np.array([arw_input, arw_input, arw_input]) * 1.0,
        # gyro bias instability, deg/hr
        'gyro_b_stability': np.array([bias_input, bias_input, bias_input]),
        # gyro bias instability correlation, sec.
        # set this to 'inf' to use a random walk model
        # set this to a positive real number to use a first-order Gauss-Markkov model
        'gyro_b_corr': np.array([100.0, 100.0, 100.0]),
        # accelerometer bias, m/s^2
        'accel_b': np.array([0.0e-3, 0.0e-3, 0.0e-3]),
        # accelerometer velocity random walk, m/s/rt-hr
        'accel_vrw': np.array([0.03119, 0.03009, 0.04779]) * 1.0,
        # accelerometer bias instability, m/s^2
        'accel_b_stability': np.array([4.29e-5, 5.72e-5, 8.02e-5]) * 1.0,
        # accelerometer bias instability correlation, sec. Similar to gyro_b_corr
        'accel_b_corr': np.array([200.0, 200.0, 200.0]),
        # magnetometer noise std, uT
        'mag_std': np.array([0.2, 0.2, 0.2]) * 1.0
    }

    odo_err = {'scale': odo_scale,
               'stdv': odo_noise}

    # create imu object
    # do not generate GPS and magnetometer data
    imu = imu_model.IMU(accuracy=imu_err, axis=9, gps=False, odo=True, odo_opt=odo_err)

    return imu


def initial_state(motion_path_filename):
    """
    This function gets the initial states from a motion profile defined in the demo_motion_def_files folder
    We can also add some initial error states in the future as needed, to simulate existing localization error
    Returns: A nine dimensional vector giving initial state of the vehicle

    """
    # Set initial positions
    initial_pos_vel_att = np.genfromtxt(motion_def_path + motion_path_filename, \
                                        delimiter=',', skip_header=1, max_rows=1)

    initial_pos_vel_att[0] = initial_pos_vel_att[0] * D2R
    initial_pos_vel_att[1] = initial_pos_vel_att[1] * D2R
    initial_pos_vel_att[6:9] = initial_pos_vel_att[6:9] * D2R

    # add initial states error if needed
    ini_vel_err = np.array([0.0, 0.0, 0.0])  # initial velocity error in the body frame, m/s
    ini_att_err = np.array([0.0, 0.0, 0.0])  # initial Euler angles error, deg
    initial_pos_vel_att[3:6] += ini_vel_err
    initial_pos_vel_att[6:9] += ini_att_err * D2R

    return initial_pos_vel_att


def create_sim_object(algorithm, motion_path_filename, fs=100, fs_gps=0.0, fs_mag=0.0, imu=imu_creation()):
    """
        'ref_frame'	Reference frame used as the navigation frame and the attitude reference.
        0: NED (default), with x axis pointing along geographic north, y axis pointing eastward, z axis pointing downward. Position will be expressed in LLA form, and the velocity of the vehicle relative to the ECEF frame will be expressed in local NED frame.
        1: a virtual inertial frame with constant g, x axis pointing along geographic or magnetic north, z axis pointing along g, y axis completing a right-handed coordinate system. Position and velocity will both be in the [x y z] form in this frame.
        **Notice: For this virtual inertial frame, position is indeed the sum of the initial position in ecef and the relative position in the virutal inertial frame. Indeed, two vectors expressed in different frames should not be added. This is done in this way here just to preserve all useful information to generate .kml files. Keep this in mind if you use this result.

        'fs'	Sample frequency of IMU, units: Hz
        'fs_gps'	Sample frequency of GNSS, units: Hz
        'fs_mag'	Sample frequency of magnetometer, units: Hz
        'time'	Time series corresponds to IMU samples, units: sec.
        'gps_time'	Time series corresponds to GNSS samples, units: sec.
        'algo_time'	Time series corresponding to algorithm output, units: ['s']. If your algorithm output data rate is different from the input data rate, you should include 'algo_time' in the algorithm output.
        'gps_visibility'	Indicate if GPS is available. 1 means yes, and 0 means no.
        'ref_pos'	True position in the navigation frame. When users choose NED (ref_frame=0) as the navigation frame, positions will be given in the form of [Latitude, Longitude, Altitude], units: ['rad', 'rad', 'm']. When users choose the virtual inertial frame, positions (initial position + positions relative to the origin of the frame) will be given in the form of [x, y, z], units: ['m', 'm', 'm'].
        'ref_vel'	True velocity w.r.t the navigation/reference frame expressed in the NED frame, units: ['m/s', 'm/s', 'm/s'].
        'ref_att_euler'	True attitude (Euler angles, ZYX rotation sequency), units: ['rad', 'rad', 'rad']
        'ref_att_quat'	True attitude (quaternions)
        'ref_gyro'	True angular velocity in the body frame, units: ['rad/s', 'rad/s', 'rad/s']
        'ref_accel'	True acceleration in the body frame, units: ['m/s^2', 'm/s^2', 'm/s^2']
        'ref_mag'	True geomagnetic field in the body frame, units: ['uT', 'uT', 'uT'] (only available when axis=9 in IMU object)
        'ref_gps'	True GPS position/velocity, ['rad', 'rad', 'm', 'm/s', 'm/s', 'm/s'] for NED (LLA), ['m', 'm', 'm', 'm/s', 'm/s', 'm/s'] for virtual inertial frame (xyz) (only available when gps=True in IMU object)
        'gyro'	Gyroscope measurements, 'ref_gyro' with errors
        'accel'	Accelerometer measurements, 'ref_accel' with errors
        'mag'	Magnetometer measurements, 'ref_mag' with errors
        'gps'	GPS measurements, 'ref_gps' with errors
        'ad_gyro'	Allan std of gyro, units: ['rad/s', 'rad/s', 'rad/s']
        'ad_accel'	Allan std of accel, units: ['m/s2', 'm/s2', 'm/s2']
        'pos'	Simulation position from algo, units: ['rad', 'rad', 'm'] for NED (LLA), ['m', 'm', 'm'] for virtual inertial frame (xyz).
        'vel'	Simulation velocity from algo, units: ['m/s', 'm/s', 'm/s']
        'att_euler'	Simulation attitude (Euler, ZYX) from algo, units: ['rad', 'rad', 'rad']
        'att_quat'	Simulation attitude (quaternion) from algo
        'wb'	Gyroscope bias estimation, units: ['rad/s', 'rad/s', 'rad/s']
        'ab'	Accelerometer bias estimation, units: ['m/s^2', 'm/s^2', 'm/s^2']
        'gyro_cal'	Calibrated gyro output, units: ['rad/s', 'rad/s', 'rad/s']
        'accel_cal'	Calibrated acceleromter output, units: ['m/s^2', 'm/s^2', 'm/s^2']
        'mag_cal'	Calibrated magnetometer output, units: ['uT', 'uT', 'uT']
        'soft_iron'	3x3 soft iron calibration matrix
        'hard_iron'	Hard iron calibration, units: ['uT', 'uT', 'uT']
    Returns:

    """

    sim = ins_sim.Sim([fs, fs_gps, fs_mag],
                      motion_def_path + motion_path_filename,
                      ref_frame=1,
                      imu=imu,
                      # vehicle maneuver capability
                      # [max accel, max angular accel, max angular rate]
                      mode=None,
                      env=None,
                      algorithm=algorithm)
    return sim


if __name__ == '__main__':
    # some major IMU parameters
    # Input some error typical for IMU380
    arw = .25
    gyro_bias = 3.5

    # Here we set the odometer error
    odo_scale = .99
    odo_noise = .1

    # Define motion path file
    motion_def_path = os.path.abspath('.\demo_motion_def_files')

    # pull over motion path
    # motion_path_filename = "\motion_def_linemerge.csv"
    # motion_path_filename = "\motion_def-90deg_turn.csv"

    # motion_path_filename = "\motion_def_linemerge.csv"
    motion_path_filename = "\motion_def_70_mph_straight_20sec.csv"
    # motion_path_filename = "\motion_def_70_mph_straight.csv"
    # motion_path_filename = "//motion_def-long_drive.csv"
    print(motion_def_path + motion_path_filename)

    # Create IMU model with parameters and initial state vector
    imu_model = imu_creation(arw_input=arw, bias_input=gyro_bias)
    initial_state_vector = initial_state(motion_path_filename)
    print(initial_state_vector)
    # choose algorithm in library (here we use multiple)
    algo1 = free_integration_odo.FreeIntegration(initial_state_vector)
    algo2 = free_integration.FreeIntegration(initial_state_vector)

    # Create the simulation object that will be simulated
    sim_object_1 = create_sim_object(algo1, motion_path_filename)
    # sim_object_2 = create_sim_object(algo2)

    # run the simulation for a number of times
    sim_object_1.run(5)
    # sim_object_2.run(10)

    # -----------------------------RESULTS SECTION--------------------------------------------
    # generate simulation results, summary
    # do not save data since the simulation runs for 1000 times and generates too many results
    sim_object_1.results('', err_stats_start=-1, gen_kml=True)
    # sim_object_2.results(err_stats_start=-1, gen_kml=True)
    # sim_object_1.plot(['ref_pos'])
#    sim.plot(['ref_vel'])
# sim_object_1.plot(['vel'], opt={'vel':'error'})
#    sim.plot(['gyro'])
    sim_object_1.plot(['pos'], opt={'pos':'error'})
#    sim.plot(['att_euler'], opt={'att_euler':'error'})
# im_object_1.plot(['pos', 'vel', 'att_euler', 'accel', 'gyro'],
#         opt={'pos':'error', 'vel':'error', 'att_euler':'error'})
