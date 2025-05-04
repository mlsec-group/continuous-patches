# -*- coding: utf-8 -*-
"""
Helper to decode binary logged sensor data from crazyflie2 with uSD-Card-Deck
"""
import argparse
from cProfile import label
from zlib import crc32
import struct
from matplotlib import scale
import numpy as np

import matplotlib.pyplot as plt
from pandas import pivot

# extract null-terminated string
def _get_name(data, idx):
    endIdx = idx
    while data[endIdx] != 0:
        endIdx = endIdx + 1
    return data[idx:endIdx].decode("utf-8"), endIdx + 1

def decode(filename):
    # read file as binary
    with open(filename, 'rb') as f:
        data = f.read()

    # check magic header
    if data[0] != 0xBC:
        print("Unsupported format!")
        return

    # check CRC
    crc = crc32(data[0:-4])
    expected_crc, = struct.unpack('I', data[-4:])
    if crc != expected_crc:
        print("WARNING: CRC does not match!")

    # check version
    version, num_event_types = struct.unpack('HH', data[1:5])
    if version != 1 and version != 2:
        print("Unsupported version!", version)
        return

    result = dict()
    event_by_id = dict()

    # read header with data types
    idx = 5
    for _ in range(num_event_types):
        event_id, = struct.unpack('H', data[idx:idx+2])
        idx += 2
        event_name, idx = _get_name(data, idx)
        result[event_name] = dict()
        result[event_name]["timestamp"] = []
        num_variables, = struct.unpack('H', data[idx:idx+2])
        idx += 2
        fmtStr = "<"
        variables = []
        for _ in range(num_variables):
            var_name_and_type, idx = _get_name(data, idx)
            var_name = var_name_and_type[0:-3]
            var_type = var_name_and_type[-2]
            result[event_name][var_name] = []
            fmtStr += var_type
            variables.append(var_name)
        event_by_id[event_id] = {
            'name': event_name,
            'fmtStr': fmtStr,
            'numBytes': struct.calcsize(fmtStr),
            'variables': variables,
            }

    while idx < len(data) - 4:
        if version == 1:
            event_id, timestamp, = struct.unpack('<HI', data[idx:idx+6])
            idx += 6
        elif version == 2:
            event_id, timestamp, = struct.unpack('<HQ', data[idx:idx+10])
            timestamp = timestamp / 1000.0
            idx += 10
        event = event_by_id[event_id]
        fmtStr = event['fmtStr']
        eventData = struct.unpack(fmtStr, data[idx:idx+event['numBytes']])
        idx += event['numBytes']
        for v,d in zip(event['variables'], eventData):
            result[event['name']][v].append(d)
        result[event['name']]["timestamp"].append(timestamp)

    # remove keys that had no data
    for event_name in list(result.keys()):
        if len(result[event_name]['timestamp']) == 0:
            del result[event_name]

    # convert to numpy arrays
    for event_name in result.keys():
        for var_name in result[event_name]:
            result[event_name][var_name] = np.array(result[event_name][var_name])

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    data = decode(args.filename)
    
    is_pos = np.array([data['fixedFrequency']['stateEstimate.x'], data['fixedFrequency']['stateEstimate.y'], data['fixedFrequency']['stateEstimate.z']])
    is_velocity = np.array([data['fixedFrequency']['stateEstimate.vx'], data['fixedFrequency']['stateEstimate.vy'], data['fixedFrequency']['stateEstimate.vz']])
    is_yaw = np.array(data['fixedFrequency']['stateEstimate.yaw'])
    
    timestamps = data['fixedFrequency']['timestamp']

    should_velocity =  np.array([data['fixedFrequency']['ctrltarget.vx'], data['fixedFrequency']['ctrltarget.vy'], data['fixedFrequency']['ctrltarget.z']])
    
    
    should_yawrate = np.array(data['fixedFrequency']['ctrltarget.yawrate']) # calculated yaw rate
    gyro_yawrate = np.array(data['fixedFrequency']['gyro.z']) # is yaw rate
    should_yaw = np.array(data['fixedFrequency']['ctrltarget.yaw']) # calculated yaw


    target_pos = np.array([data['fixedFrequency']['frontnet.targetx'], data['fixedFrequency']['frontnet.targety'], data['fixedFrequency']['frontnet.targetz']])
    target_yaw = np.array(data['fixedFrequency']['frontnet.targetyaw'])


    # plotting the pose + velocities 
    fig1 = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot3D(*is_pos, 'red')
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_zlabel('Z-Axis')

    for i, velocity in enumerate(zip(*is_velocity)):

        if i % 30 == 0:
            vlength = np.linalg.norm(velocity)
            ax.quiver(is_pos[0, i], is_pos[1, i], is_pos[2, i],  *velocity, 
                    pivot='tail', length=vlength, arrow_length_ratio=0.1)

    
    # plotting x, y and z w.r.t. the the time
    fig2, ax2 = plt.subplots(3, 1)
    ax2[0].set_title('x')
    ax2[0].plot(timestamps, is_pos[0])
    ax2[1].set_title('y')
    ax2[1].plot(timestamps, is_pos[1])
    ax2[2].set_title('z')
    ax2[2].plot(timestamps, is_pos[2])
    ax2[2].set_xlabel('time')


    # velocity plots not needed currently 
    # fig3, ax3 = plt.subplots(3, 1)
    # ax3[0].set_title('vx')
    # ax3[0].plot(timestamps, is_velocity[0], label='is')
    # ax3[0].plot(timestamps, should_velocity[0], label='desired')
    # ax3[1].set_title('vy')
    # ax3[1].plot(timestamps, is_velocity[1], label='is')
    # ax3[1].plot(timestamps, should_velocity[1], label='desired')
    # ax3[2].set_title('z')
    # ax3[2].plot(timestamps, is_pos[2], label='is')
    # ax3[2].plot(timestamps, should_velocity[2], label='desired')
    # ax2[2].set_xlabel('time')
    # ax3[0].legend()
    # ax3[1].legend()
    # ax3[2].legend()

    # plot yaw angles of current and target UAV
    fig4, ax4 = plt.subplots(1, 1)
    ax4.set_title('yaw')
    ax4.plot(timestamps, np.abs(np.radians(is_yaw)), label='current')
    ax4.plot(timestamps, np.abs(np.radians(should_yaw)), label='calc')
    ax4.plot(timestamps, target_yaw, label='target')
    ax4.legend()


    plt.show()
