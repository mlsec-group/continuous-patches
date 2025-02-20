
import time
from flying.cf_control import CrazyflieControl, custom_sleep
import yaml
from util import PoseUpdater
from Display import PatchDisplayThread
import numpy as np

with open('flying/config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
print(config)

display_updater = PatchDisplayThread('Patch', (2561, 0))

projector_display_size = (1050, 1680)
background = np.zeros((*projector_display_size, 3), dtype=np.uint8)

display_updater.start()
display_updater.update_simple(background)

cf = CrazyflieControl(config)

pose_updater = PoseUpdater(cf.pose)


patch = np.random.rand(80, 80, 3) * 255

sf_opt = 1.
tx_opt = 0.5 
ty_opt = 0.5

while True:
    try:
        current_pose, _ = pose_updater.get_current_pose()
        # print("Current pose:", current_pose)
        display_updater.update(patch, sf_opt, tx_opt, ty_opt, current_pose)
        time.sleep(0.1)
    except KeyboardInterrupt:
        break

display_updater.close()
pose_updater.close()
cf.close()



# bat_v, bat_s = cf.battery

# while bat_v is None:
#     bat_v, bat_s = cf.battery

# print(f"Battery voltage: {bat_v}, Battery state: {bat_s}")

# if bat_v <= 3900: 
#     print('Battery below 3.9V!! Not flying.')
#     cf.close()
#     # display_thread.close()
#     exit()

# cf.takeoff(1.0, 3)

# custom_sleep(2., cf.occupied)


# cf.toggle_frontnet()

# custom_sleep(10., cf.occupied, True)

# cf.toggle_frontnet()
# custom_sleep(2., cf.occupied, True)

# cf.reset()
# custom_sleep(2., cf.occupied, True)

# cf.land()
# custom_sleep(2., cf.occupied, True)
# cf.close()
# exit()