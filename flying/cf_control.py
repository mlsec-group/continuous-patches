import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.utils import uri_helper
from cflib.crazyflie.log import LogConfig
from cflib.utils.power_switch import PowerSwitch

from cflib.crazyflie.syncLogger import SyncLogger

import cflib.crtp
from cflib.crtp.crtpstack import CRTPPacket
from cflib.crtp.crtpstack import CRTPPort

import time
import numpy as np

from threading import Thread

import motioncapture

### code from https://github.com/bitcraze/crazyflie-lib-python/blob/master/examples/mocap/mocap_hl_commander.py
# When using full pose, the estimator can be sensitive to noise in the orientation data when yaw is close to +/- 90
# degrees. If this is a problem, increase orientation_std_dev a bit. The default value in the firmware is 4.5e-3.
orientation_std_dev = 8.0e-3

# True: send position and orientation; False: send position only
send_full_pose = True


class MocapWrapper(Thread):
    def __init__(self, body_name, mocap_system_type, host_name):
        Thread.__init__(self)

        self.body_name = body_name
        self.mocap_system_type = mocap_system_type
        self.host_name = host_name
        self.on_pose = None
        self._stay_open = True

        self.start()

    def close(self):
        self._stay_open = False

    def run(self):
        mc = motioncapture.connect(self.mocap_system_type, {'hostname': self.host_name})
        while self._stay_open:
            mc.waitForNextFrame()
            for name, obj in mc.rigidBodies.items():
                if name == self.body_name:
                    if self.on_pose:
                        pos = obj.position
                        self.on_pose([pos[0], pos[1], pos[2], obj.rotation])

def wait_for_position_estimator(scf):
    print('Waiting for estimator to find position...')

    log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.001

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_x_history.append(data['kalman.varPX'])
            var_x_history.pop(0)
            var_y_history.append(data['kalman.varPY'])
            var_y_history.pop(0)
            var_z_history.append(data['kalman.varPZ'])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            # print("{} {} {}".
            #       format(max_x - min_x, max_y - min_y, max_z - min_z))

            if (max_x - min_x) < threshold and (
                    max_y - min_y) < threshold and (
                    max_z - min_z) < threshold:
                break

def send_extpose_quat(cf, x, y, z, quat):
    """
    Send the current Crazyflie X, Y, Z position and attitude as a quaternion.
    This is going to be forwarded to the Crazyflie's position estimator.
    """
    if send_full_pose:
        cf.extpos.send_extpose(x, y, z, quat.x, quat.y, quat.z, quat.w)
    else:
        cf.extpos.send_extpos(x, y, z)


def reset_estimator(cf):
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')

    # time.sleep(1)
    wait_for_position_estimator(cf)


def adjust_orientation_sensitivity(cf):
    cf.param.set_value('locSrv.extQuatStdDev', orientation_std_dev)


def activate_kalman_estimator(cf):
    cf.param.set_value('stabilizer.estimator', '2')

    # Set the std deviation for the quaternion data pushed into the
    # kalman filter. The default value seems to be a bit too low.
    cf.param.set_value('locSrv.extQuatStdDev', 0.06)

### end of code from Bitcraze

class CrazyflieControl():
    def __init__(self, uri):#, positions_queue):
        self.connected= False
        self.uri = uri_helper.uri_from_env(default=uri)
        # self.positions_queue = positions_queue
        # print(self.positions_queue)

        print("Init CF drivers...")
        cflib.crtp.init_drivers()

        print("Rebooting..")
        self.reboot()

        print("Init Mocap thread..")
        self.mocap_wrapper = MocapWrapper('cf_pia', 'optitrack', '141.23.110.143')

        self.connect()
        self.init_logger()
        self.pose = None
        self.battery = [None, None]
        #self.lighthouse = 0.

        

    def connect(self):
        self.cf = Crazyflie(rw_cache='./cache')
        self.scf = SyncCrazyflie(self.uri, cf=self.cf)
        self.scf.open_link()
        self.connected = self.scf.cf.fully_connected

        # Set up a callback to handle data from the mocap system
        self.mocap_wrapper.on_pose = lambda pose: send_extpose_quat(self.scf.cf, pose[0], pose[1], pose[2], pose[3])
        adjust_orientation_sensitivity(self.scf.cf)
        activate_kalman_estimator(self.scf.cf)
        reset_estimator(self.scf.cf)


        self.commander = self.scf.cf.high_level_commander

    def init_logger(self):
        self._lg_stab = LogConfig(name='Stabilizer', period_in_ms=10)
        self._lg_stab.add_variable('stateEstimate.x', 'float')
        self._lg_stab.add_variable('stateEstimate.y', 'float')
        self._lg_stab.add_variable('stateEstimate.z', 'float')
        self._lg_stab.add_variable('stabilizer.yaw', 'float')   # yaw in degrees

        self._lg_stat = LogConfig(name='Status', period_in_ms=1000)
        self._lg_stat.add_variable('pm.vbatMV', 'uint16_t')
        self._lg_stat.add_variable('pm.state', 'uint8_t')
        self._lg_stat.add_variable('lighthouse.posRt', 'float')

        try:
            print("Adding config")
            self.cf.log.add_config(self._lg_stab)
            self.cf.log.add_config(self._lg_stat)
            # This callback will receive the data
            self._lg_stab.data_received_cb.add_callback(self._stab_log_data)
            self._lg_stat.data_received_cb.add_callback(self._status_log_data)
            # This callback will be called on errors
            self._lg_stab.error_cb.add_callback(self._stab_log_error)
            self._lg_stat.error_cb.add_callback(self._stab_log_error)

            # Start the logging
            print("Start logging...")
            self._lg_stab.start()
            self._lg_stat.start()
            print("Logger configured!")
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Stabilizer log config, bad configuration.')


    def _stab_log_error(self, logconf, msg):
        """Callback from the log API when an error occurs"""
        print('Error when logging %s: %s' % (logconf.name, msg))

    def _stab_log_data(self, timestamp, data, logconf):
        """Callback from a the log API when data arrives"""
        # print(f'[{timestamp}][{logconf.name}]: ', end='')
        # for name, value in data.items():
        #     print(f'{name}: {value:3.3f} ', end='')
        # print()
        self.pose = np.array(list(data.values()))

    def _status_log_data(self, timestamp, data, logconf):
        self.battery = list(data.values())[:2]
        self.lighthouse = list(data.values())[-1]

    def is_stm_connected(self):
        link = cflib.crtp.get_link_driver(self.uri)

        pk = CRTPPacket()
        pk.set_header(CRTPPort.LINKCTRL, 0)  # Echo channel
        pk.data = b'test'
        link.send_packet(pk)
        for _ in range(10):
            pk_ack = link.receive_packet(0.1)
            # print(pk_ack)
            if pk_ack is not None and pk.data == pk_ack.data:
                link.close()
                return True
        link.close()
        return False

    def takeoff(self):
        print("taking off..")
        self.commander.takeoff(0.6, 2.0)
        # time.sleep(5.0)

    def land(self):
        self.commander.land(0.0, 2.0)
        # time.sleep(2.)
        # self.commander.stop()

    def angular_distance(self, x, y):
        return np.arctan2(np.sin(x-y), np.cos(x-y))

    def goto_auto(self, x, y, z, yaw=0.):
        dist = np.linalg.norm(self.pose[:3]-np.array([x, y, z]))
        # print("auto pose: ", self.pose)
        angular_dist = np.abs(self.angular_distance(np.radians(self.pose[3]), np.radians(yaw)))
        # print("auto angular dist: ", angular_dist)
        angular_speed = 1.0
        speed = 0.6
        t = dist/speed
        time_angular = angular_dist/angular_speed
        # print("auto time: ", t, time_angular)
        self.goto(x, y, z, yaw, max(t, time_angular))
        # time.sleep(max(max(t, time_angular)+sleep_delta,0.2))
        return max(t, time_angular)

    def goto(self, x, y, z, yaw=0.0, seconds=2.0):
        self.commander.go_to(x, y, z, np.radians(yaw), seconds)     # yaw in rad

    def power_off(self):
        s = PowerSwitch(self.uri)
        # s.stm_power_cycle()
        s.stm_power_down()

    def reboot(self):
        # self.connected = False
        s = PowerSwitch(self.uri)
        # s.stm_power_cycle()
        s.stm_power_down()
        time.sleep(3)
        s.stm_power_up()
        s.close()
        time.sleep(2.)
        while not self.is_stm_connected():
            time.sleep(0.1)
        time.sleep(4.)
        print("...CF rebooted!")
        self.connected = True

    def close(self):
        self.scf.close()
        self.mocap_wrapper.close()
        self.cf.close()
        # self.power_off()



if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Script to start flying your CF and simultanously stream images from the AI Deck')
    # parser.add_argument('--uri', default="radio://0/80/2M/E7E7E7E7E7", help="Crazyflie URI")
    # parser.add_argument('--ssid', default="TP-Link_67BC", help="WiFi SSID")
    # parser.add_argument('--ip', default="192.168.0.100", help="AI Deck IP")
    # parser.add_argument('--port', type=int, default=5000, help="AI Deck IP")
    # args = parser.parse_args()



    cf = CrazyflieControl("radio://0/80/2M/E7E7E7E712")
    cf.takeoff()
    time.sleep(5.)
    cf.land()
    cf.close()
