from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import numpy as np
import random
import collections as col
import tensorflow as tf
import re
from itertools import chain
import ReplayBuffer

# The sys module basically provides access to some variables used or maintained by the interpreter
# and to functions that interact strongly with the interpreter e.g. sys.path.append(path) includes the
# path in the search path for the modules

# The OS module in Python provides a way of using operating system dependent
# functionality
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo "
        "installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci

tf.reset_default_graph()

ACTIONS = ['Lane1L', 'Lane1S', 'Lane2L', 'Lane2S', 'Lane3L', 'Lane3S', 'Lane4L', 'Lane4S']
N_ACTIONS = len(ACTIONS)
LIST_STATES = []

EPSILON_Orig = 0.7 # Greedy Policy
ALPHA = 0.0001 # Learning Rate
GAMMA = 0.9 # Discount Factor

MAX_EPISODES = 30
MAX_STEPS = 4000
MAX_EXPLORATION_STEPS = 60000
step = 0

PRE_TRAIN_STEPS = 45000
update_frequency = 5000
batch_size = 64

np.random.seed(1337)

input_NN = tf.placeholder(shape=[1, 8], dtype=tf.float64)

weights1 = tf.Variable(tf.random_uniform([8, 16], 0, 0.01, dtype=tf.float64))
bias1 = tf.Variable(tf.zeros([16], dtype=tf.float64))
output1 = tf.matmul(input_NN, weights1) + bias1

weights2 = tf.Variable(tf.random_uniform([16, 8], 0, 0.01, dtype=tf.float64))
Qout = tf.nn.softmax(tf.matmul(output1, weights2))
predict = tf.argmax(Qout, 1)

nextQ = tf.placeholder(shape=[1, 8], dtype=tf.float64)
loss = tf.square(nextQ - Qout)
trainer = tf.train.AdadeltaOptimizer(learning_rate=ALPHA)
updateModel = trainer.minimize(loss)

global managed_vehicles
managed_vehicles = []

def playGame():

    global managed_vehicles

    # SUMO STUFF

    options = get_options()

    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    init = tf.initialize_all_variables()
    EPSILON = EPSILON_Orig

    global_step = 0

    myBuffer = ReplayBuffer.experience_buffer()

    with tf.Session() as sess:
        sess.run(init)

        for episode in range (MAX_EPISODES):
		
			if episode < MAX_EPISODES - 1:
                train = True
                test = False
            else:
                train = False
                test = True

            episodeBuffer = ReplayBuffer.experience_buffer()

            vehicles_waiting_1_S = []
            vehicles_waiting_1_L = []
            vehicles_waiting_2_S = []
            vehicles_waiting_2_L = []
            vehicles_waiting_3_S = []
            vehicles_waiting_3_L = []
            vehicles_waiting_4_S = []
            vehicles_waiting_4_L = []

            vehicles_in_loop_1_S = []
            vehicles_in_loop_1_L = []
            vehicles_in_loop_2_S = []
            vehicles_in_loop_2_L = []
            vehicles_in_loop_3_S = []
            vehicles_in_loop_3_L = []
            vehicles_in_loop_4_S = []
            vehicles_in_loop_4_L = []

            position1_S = []
            position1_L = []
            position2_S = []
            position2_L = []
            position3_S = []
            position3_L = []
            position4_S = []
            position4_L = []

            running_vehicle_ids = []
            lane_change_disabled_vehicles = []

            collided_vehicles = []

            generate_routefile()

            waiting_time = 0
			
			total_reward = 0

            Tot_wt_time = 0

            total_co2_emission = 0

            traci.start([sumoBinary, "-c", "data/cross.sumocfg",
                         "--tripinfo-output", "tripinfo.xml", "--additional-files",
                         "data/cross.additionals.xml", "--collision.check-junctions",
                         "--collision.action", "warn", "--step-length", "1",
                         "--error-log", "error.txt"])

            S = np.zeros([1, 8], dtype=float)
            A = 'Lane1L'
            B = np.zeros(1)

            Action = []

            for step in range (MAX_STEPS):
			
				if train:

					R_ = 0
					R = 0
					R_collision = 0
					wt_time_1L = 0
					wt_time_1S = 0
					wt_time_2L = 0
					wt_time_2S = 0
					wt_time_3L = 0
					wt_time_3S = 0
					wt_time_4L = 0
					wt_time_4S = 0

					wt_1L = 0
					wt_1S = 0
					wt_2L = 0
					wt_2S = 0
					wt_3L = 0
					wt_3S = 0
					wt_4L = 0
					wt_4S = 0

					Tot_co2_1L = 0
					Tot_co2_1S = 0
					Tot_co2_2L = 0
					Tot_co2_2S = 0
					Tot_co2_3L = 0
					Tot_co2_3S = 0
					Tot_co2_4L = 0
					Tot_co2_4S = 0
					list_is_new = False

					if len(running_vehicle_ids) > 300:
						running_vehicle_ids = []
					if len(lane_change_disabled_vehicles) > 300:
						lane_change_disabled_vehicles = []

					running_vehicle_ids = traci.vehicle.getIDList()
					for vehicleID in running_vehicle_ids:
						if vehicleID not in lane_change_disabled_vehicles:
							traci.vehicle.setSpeedMode(vehicleID, 0b00000)
							traci.vehicle.setLaneChangeMode(vehicleID, 0b0100000000)
							lane_change_disabled_vehicles.append(vehicleID)

					S_old = S.copy()
					S_col = S_old.copy()
					A_old = A
					B_old = B.copy()

					B, allQ = sess.run([predict, Qout], feed_dict={input_NN:S})

					S_ = S.copy()

					if B == 0:
						A = 'Lane1L'
					if B == 1:
						A = 'Lane1S'
					if B == 2:
						A = 'Lane2L'
					if B == 3:
						A = 'Lane2S'
					if B == 4:
						A = 'Lane3L'
					if B == 5:
						A = 'Lane3S'
					if B == 6:
						A = 'Lane4L'
					if B == 7:
						A = 'Lane4S'

					if np.random.uniform() > EPSILON:
						if (vehicles_waiting_1_S or vehicles_waiting_1_L or
								vehicles_waiting_2_L or vehicles_waiting_2_S or
								vehicles_waiting_3_S or vehicles_waiting_3_L or
								vehicles_waiting_4_S or vehicles_waiting_4_L):
							while(1):
								ran_int = np.random.randint(0, 8)

								if ran_int == 0:
									if vehicles_waiting_1_L:
										A = 'Lane1L'
										break

								if ran_int == 1:
									if vehicles_waiting_1_S:
										A = 'Lane1S'
										break

								if ran_int == 2:
									if vehicles_waiting_2_L:
										A = 'Lane2L'
										break

								if ran_int == 3:
									if vehicles_waiting_2_S:
										A = 'Lane2S'
										break

								if ran_int == 4:
									if vehicles_waiting_3_L:
										A = 'Lane3L'
										break

								if ran_int == 5:
									if vehicles_waiting_3_S:
										A = 'Lane3S'
										break

								if ran_int == 6:
									if vehicles_waiting_4_L:
										A = 'Lane4L'
										break

								if ran_int == 7:
									if vehicles_waiting_4_S:
										A = 'Lane4S'
										break
						else:
							A = np.random.choice(ACTIONS)


					if (A == 'Lane1L' or ((S[0, 0] > 0) and np.equal(S[0, 1], 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
										  and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))
							or ((S[0, 0] > 0) and (S[0, 1] > 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
								and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))):

						if vehicles_waiting_1_L:
							try:
								traci.vehicle.setStop(vehicles_waiting_1_L[0], "1i", position1_L[0], 2, 0)
								traci.vehicle.setSpeed(vehicles_waiting_1_L[0], 10)
								traci.vehicle.setSpeedMode(vehicles_waiting_1_L[0], 0)
								wt_time_1L = traci.vehicle.getWaitingTime(vehicles_waiting_1_L[0])
								vehicles_waiting_1_L.pop(0)
								position1_L.pop(0)
								A = 'Lane1L'
								B[0] = 0
							except Exception:
								traci.vehicle.remove(vehicles_waiting_1_L[0], reason=3)
								if vehicles_waiting_1_L[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_1_L[0])
								vehicles_in_loop_1_L.remove(vehicles_waiting_1_L[0])
								if vehicles_waiting_1_L[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_1_L[0])
								if vehicles_waiting_1_L[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_1_L[0])
								vehicles_waiting_1_L.pop(0)
								position1_L.pop(0)


					if (A == 'Lane1S' or (np.equal(S[0, 0], 0) and (S[0, 1] > 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
										  and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))
							or ((S[0, 0] > 0) and (S[0, 1] > 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
								and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))):

						if vehicles_waiting_1_S:
							try:
								traci.vehicle.setStop(vehicles_waiting_1_S[0], "1i", position1_S[0], 1, 0)
								traci.vehicle.setSpeed(vehicles_waiting_1_S[0], 10)
								traci.vehicle.setSpeedMode(vehicles_waiting_1_S[0], 0)
								wt_time_1S = traci.vehicle.getWaitingTime(vehicles_waiting_1_S[0])
								vehicles_waiting_1_S.pop(0)
								position1_S.pop(0)
								A = 'Lane1S'
								B[0] = 1
							except Exception:
								traci.vehicle.remove(vehicles_waiting_1_S[0], reason=3)
								if vehicles_waiting_1_S[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_1_S[0])
								vehicles_in_loop_1_S.remove(vehicles_waiting_1_S[0])
								if vehicles_waiting_1_S[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_1_S[0])
								if vehicles_waiting_1_S[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_1_S[0])
								vehicles_waiting_1_S.pop(0)
								position1_S.pop(0)


					if (A == 'Lane2L' or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and (S[0, 2] > 0) and np.equal(S[0, 3], 0)
										  and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))
							or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and (S[0, 2] > 0) and (S[0, 3] > 0)
								and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))):

						if vehicles_waiting_2_L:
							try:
								traci.vehicle.setStop(vehicles_waiting_2_L[0], "2i", position2_L[0], 2, 0)
								traci.vehicle.setSpeed(vehicles_waiting_2_L[0], 10)
								traci.vehicle.setSpeedMode(vehicles_waiting_2_L[0], 0)
								wt_time_2L = traci.vehicle.getWaitingTime(vehicles_waiting_2_L[0])
								vehicles_waiting_2_L.pop(0)
								position2_L.pop(0)
								A = 'Lane2L'
								B[0] = 2
							except Exception:
								traci.vehicle.remove(vehicles_waiting_2_L[0], reason=3)
								if vehicles_waiting_2_L[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_2_L[0])
								vehicles_in_loop_2_L.remove(vehicles_waiting_2_L[0])
								if vehicles_waiting_2_L[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_2_L[0])
								if vehicles_waiting_2_L[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_2_L[0])
								vehicles_waiting_2_L.pop(0)
								position2_L.pop(0)


					if (A == 'Lane2S' or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and np.equal(S[0, 2], 0) and (S[0, 3] > 0)
										  and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))
							or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and (S[0, 2] > 0) and (S[0, 3] > 0)
								and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))):

						if vehicles_waiting_2_S:
							try:
								traci.vehicle.setStop(vehicles_waiting_2_S[0], "2i", position2_S[0], 1, 0)
								traci.vehicle.setSpeed(vehicles_waiting_2_S[0], 10)
								traci.vehicle.setSpeedMode(vehicles_waiting_2_S[0], 0)
								wt_time_2S = traci.vehicle.getWaitingTime(vehicles_waiting_2_S[0])
								vehicles_waiting_2_S.pop(0)
								position2_S.pop(0)
								A = 'Lane2S'
								B[0] = 3
							except Exception:
								traci.vehicle.remove(vehicles_waiting_2_S[0], reason=3)
								if vehicles_waiting_2_S[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_2_S[0])
								vehicles_in_loop_2_S.remove(vehicles_waiting_2_S[0])
								if vehicles_waiting_2_S[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_2_S[0])
								if vehicles_waiting_2_S[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_2_S[0])
								vehicles_waiting_2_S.pop(0)
								position2_S.pop(0)


					if (A == 'Lane3L' or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
										  and (S[0, 4] > 0) and np.equal(S[0, 5], 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))
							or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
								and (S[0, 4] > 0) and (S[0, 5] > 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))):

						if vehicles_waiting_3_L:
							try:
								traci.vehicle.setStop(vehicles_waiting_3_L[0], "3i", position3_L[0], 2, 0)
								traci.vehicle.setSpeed(vehicles_waiting_3_L[0], 10)
								traci.vehicle.setSpeedMode(vehicles_waiting_3_L[0], 0)
								wt_time_3L = traci.vehicle.getWaitingTime(vehicles_waiting_3_L[0])
								vehicles_waiting_3_L.pop(0)
								position3_L.pop(0)
								A = 'Lane3L'
								B[0] = 4
							except Exception:
								traci.vehicle.remove(vehicles_waiting_3_L[0], reason=3)
								if vehicles_waiting_3_L[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_3_L[0])
								vehicles_in_loop_3_L.remove(vehicles_waiting_3_L[0])
								if vehicles_waiting_3_L[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_3_L[0])
								if vehicles_waiting_3_L[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_3_L[0])
								vehicles_waiting_3_L.pop(0)
								position3_L.pop(0)

					if (A == 'Lane3S' or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
										  and np.equal(S[0, 4], 0) and (S[0, 5] > 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))
							or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
								and (S[0, 4] > 0) and (S[0, 5] > 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))):

						if vehicles_waiting_3_S:
							try:
								traci.vehicle.setStop(vehicles_waiting_3_S[0], "3i", position3_S[0], 1, 0)
								traci.vehicle.setSpeed(vehicles_waiting_3_S[0], 10)
								traci.vehicle.setSpeedMode(vehicles_waiting_3_S[0], 0)
								wt_time_3S = traci.vehicle.getWaitingTime(vehicles_waiting_3_S[0])
								vehicles_waiting_3_S.pop(0)
								position3_S.pop(0)
								A = 'Lane3S'
								B[0] = 5
							except:
								traci.vehicle.remove(vehicles_waiting_3_S[0], reason=3)
								if vehicles_waiting_3_S[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_3_S[0])
								vehicles_in_loop_3_S.remove(vehicles_waiting_3_S[0])
								if vehicles_waiting_3_S[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_3_S[0])
								if vehicles_waiting_3_S[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_3_S[0])
								vehicles_waiting_3_S.pop(0)
								position3_S.pop(0)


					if (A == 'Lane4L' or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
										  and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and (S[0, 6] > 0) and np.equal(S[0, 7], 0))
							or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
								and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and (S[0, 6] > 0) and (S[0, 7] > 0))):

						if vehicles_waiting_4_L:
							try:
								traci.vehicle.setStop(vehicles_waiting_4_L[0], "4i", position4_L[0], 2, 0)
								traci.vehicle.setSpeed(vehicles_waiting_4_L[0], 10)
								traci.vehicle.setSpeedMode(vehicles_waiting_4_L[0], 0)
								wt_time_4L = traci.vehicle.getWaitingTime(vehicles_waiting_4_L[0])
								vehicles_waiting_4_L.pop(0)
								position4_L.pop(0)
								A = 'Lane4L'
								B[0] = 6
							except Exception:
								traci.vehicle.remove(vehicles_waiting_4_L[0], reason=3)
								if vehicles_waiting_4_L[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_4_L[0])
								vehicles_in_loop_4_L.remove(vehicles_waiting_4_L[0])
								if vehicles_waiting_4_L[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_4_L[0])
								if vehicles_waiting_4_L[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_4_L[0])
								vehicles_waiting_4_L.pop(0)
								position4_L.pop(0)


					if (A == 'Lane4S' or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
										  and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and np.equal(S[0, 6], 0) and (S[0, 7] > 0))
							or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
								and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and (S[0, 6] > 0) and (S[0, 7] > 0))):

						if vehicles_waiting_4_S:
							try:
								traci.vehicle.setStop(vehicles_waiting_4_S[0], "4i", position4_S[0], 1, 0)
								traci.vehicle.setSpeed(vehicles_waiting_4_S[0], 10)
								traci.vehicle.setSpeedMode(vehicles_waiting_4_S[0], 0)
								wt_time_4S = traci.vehicle.getWaitingTime(vehicles_waiting_4_S[0])
								vehicles_waiting_4_S.pop(0)
								position4_S.pop(0)
								A = 'Lane4S'
								B[0] = 7
							except Exception:
								traci.vehicle.remove(vehicles_waiting_4_S[0], reason=3)
								if vehicles_waiting_4_S[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_4_S[0])
								vehicles_in_loop_4_S.remove(vehicles_waiting_4_S[0])
								if vehicles_waiting_4_S[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_4_S[0])
								if vehicles_waiting_4_S[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_4_S[0])
								vehicles_waiting_4_S.pop(0)
								position4_S.pop(0)


					if 485 not in position1_S:
						if vehicles_waiting_1_S:
							orig_position1_S = position1_S.copy()
							pos1_S = 485
							i1S = 0
							except1_S = 0
							change_position1_S = []
							for vehicle1_S in vehicles_waiting_1_S:
								try:
									traci.vehicle.setStop(vehicle1_S, "1i", position1_S[i1S], 1, 0)
									traci.vehicle.setStop(vehicle1_S, "1i", pos1_S, 1)
									change_position1_S.append(pos1_S)
									pos1_S -= 3
									i1S += 1
								except Exception:
									except1_S = 1
									continue
							if except1_S:
								traci.vehicle.remove(vehicles_waiting_1_S[0], reason=3)
								if vehicles_waiting_1_S[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_1_S[0])
								vehicles_in_loop_1_S.remove(vehicles_waiting_1_S[0])
								if vehicles_waiting_1_S[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_1_S[0])
								if vehicles_waiting_1_S[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_1_S[0])
								vehicles_waiting_1_S.pop(0)
								position1_S.pop(0)
							else:
								position1_S = change_position1_S

					if 485 not in position1_L:
						if vehicles_waiting_1_L:
							pos1_L = 485
							i1L = 0
							orig_position1_L = position1_L.copy()
							except1_L = 0
							change_position1_L = []
							for vehicle1_L in vehicles_waiting_1_L:
								try:
									traci.vehicle.setStop(vehicle1_L, "1i", position1_L[i1L], 2, 0)
									traci.vehicle.setStop(vehicle1_L, "1i", pos1_L, 2)
									change_position1_L.append(pos1_L)
									pos1_L -= 3
									i1L += 1
								except Exception:
									except1_L = 1
									continue
							if except1_L:
								traci.vehicle.remove(vehicles_waiting_1_L[0], reason=3)
								if vehicles_waiting_1_L[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_1_L[0])
								vehicles_in_loop_1_L.remove(vehicles_waiting_1_L[0])
								if vehicles_waiting_1_L[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_1_L[0])
								if vehicles_waiting_1_L[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_1_L[0])
								vehicles_waiting_1_L.pop(0)
								position1_L.pop(0)
							else:
								position1_L = change_position1_L

					if 485 not in position2_S:
						if vehicles_waiting_2_S:
							pos2_S = 485
							i2S = 0
							except2_S = 0
							orig_position2_S = position2_S.copy()
							change_position2_S = []
							for vehicle2_S in vehicles_waiting_2_S:
								try:
									traci.vehicle.setStop(vehicle2_S, "2i", position2_S[i2S], 1, 0)
									traci.vehicle.setStop(vehicle2_S, "2i", pos2_S, 1)
									change_position2_S.append(pos2_S)
									pos2_S -= 3
									i2S += 1
								except Exception:
									except2_S = 1
									continue
							if except2_S:
								traci.vehicle.remove(vehicles_waiting_2_S[0], reason=3)
								if vehicles_waiting_2_S[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_2_S[0])
								vehicles_in_loop_2_S.remove(vehicles_waiting_2_S[0])
								if vehicles_waiting_2_S[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_2_S[0])
								if vehicles_waiting_2_S[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_2_S[0])
								vehicles_waiting_2_S.pop(0)
								position2_S.pop(0)
							else:
								position2_S = change_position2_S

					if 485 not in position2_L:
						if vehicles_waiting_2_L:
							pos2_L = 485
							i2L = 0
							orig_position2_L = position2_L.copy()
							except2_L = 0
							change_position2_L = []
							for vehicle2_L in vehicles_waiting_2_L:
								try:
									traci.vehicle.setStop(vehicle2_L, "2i", position2_L[i2L], 2, 0)
									traci.vehicle.setStop(vehicle2_L, "2i", pos2_L, 2)
									change_position2_L.append(pos2_L)
									pos2_L -= 3
									i2L += 1
								except Exception:
									except2_L = 1
									continue
							if except2_L:
								traci.vehicle.remove(vehicles_waiting_2_L[0], reason=3)
								if vehicles_waiting_2_L[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_2_L[0])
								vehicles_in_loop_2_L.remove(vehicles_waiting_2_L[0])
								if vehicles_waiting_2_L[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_2_L[0])
								if vehicles_waiting_2_L[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_2_L[0])
								vehicles_waiting_2_L.pop(0)
								position2_L.pop(0)
							else:
								position2_L = change_position2_L

					if 485 not in position3_S:
						if vehicles_waiting_3_S:
							pos3_S = 485
							i3S = 0
							orig_position3_S = position3_S.copy()
							except3_S = 0
							change_position3_S = []
							for vehicle3_S in vehicles_waiting_3_S:
								try:
									traci.vehicle.setStop(vehicle3_S, "3i", position3_S[i3S], 1, 0)
									traci.vehicle.setStop(vehicle3_S, "3i", pos3_S, 1)
									change_position3_S.append(pos3_S)
									pos3_S -= 3
									i3S += 1
								except Exception:
									except3_S = 1
									continue
							if except3_S:
								traci.vehicle.remove(vehicles_waiting_3_S[0], reason=3)
								if vehicles_waiting_3_S[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_3_S[0])
								vehicles_in_loop_3_S.remove(vehicles_waiting_3_S[0])
								if vehicles_waiting_3_S[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_3_S[0])
								if vehicles_waiting_3_S[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_3_S[0])
								vehicles_waiting_3_S.pop(0)
								position3_S.pop(0)
							else:
								position3_S = change_position3_S

					if 485 not in position3_L:
						if vehicles_waiting_3_L:
							pos3_L = 485
							i3L = 0
							orig_position3_L = position3_L.copy()
							except3_L = 0
							change_position3_L = []
							for vehicle3_L in vehicles_waiting_3_L:
								try:
									traci.vehicle.setStop(vehicle3_L, "3i", position3_L[i3L], 2, 0)
									traci.vehicle.setStop(vehicle3_L, "3i", pos3_L, 2)
									change_position3_L.append(pos3_L)
									pos3_L -= 3
									i3L += 1
								except Exception:
									except3_L = 1
									continue
							if except3_L:
								traci.vehicle.remove(vehicles_waiting_3_L[0], reason=3)
								if vehicles_waiting_3_L[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_3_L[0])
								vehicles_in_loop_3_L.remove(vehicles_waiting_3_L[0])
								if vehicles_waiting_3_L[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_3_L[0])
								if vehicles_waiting_3_L[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_3_L[0])
								vehicles_waiting_3_L.pop(0)
								position3_L.pop(0)
							else:
								position3_L = change_position3_L

					if 485 not in position4_S:
						if vehicles_waiting_4_S:
							pos4_S = 485
							i4S = 0
							except4_S = 0
							orig_position4_S = position4_S.copy()
							change_position4_S = []
							for vehicle4_S in vehicles_waiting_4_S:
								try:
									traci.vehicle.setStop(vehicle4_S, "4i", position4_S[i4S], 1, 0)
									traci.vehicle.setStop(vehicle4_S, "4i", pos4_S, 1)
									change_position4_S.append(pos4_S)
									pos4_S -= 3
									i4S += 1
								except Exception:
									except4_S = 1
									continue
							if except4_S:
								traci.vehicle.remove(vehicles_waiting_4_S[0], reason=3)
								if vehicles_waiting_4_S[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_4_S[0])
								vehicles_in_loop_4_S.remove(vehicles_waiting_4_S[0])
								if vehicles_waiting_4_S[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_4_S[0])
								if vehicles_waiting_4_S[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_4_S[0])
								vehicles_waiting_4_S.pop(0)
								position4_S.pop(0)
							else:
								position4_S = change_position4_S

					if 485 not in position4_L:
						if vehicles_waiting_4_L:
							pos4_L = 485
							i4L = 0
							orig_position4_L = position4_L.copy()
							except4_L = 0
							change_position4_L = []
							for vehicle4_L in vehicles_waiting_4_L:
								try:
									traci.vehicle.setStop(vehicle4_L, "4i", position4_L[i4L], 2, 0)
									traci.vehicle.setStop(vehicle4_L, "4i", pos4_L, 2)
									change_position4_L.append(pos4_L)
									pos4_L -= 3
									i4L += 1
								except Exception:
									except4_L = 1
									continue
							if except4_L:
								traci.vehicle.remove(vehicles_waiting_4_L[0], reason=3)
								if vehicles_waiting_4_L[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_4_L[0])
								vehicles_in_loop_4_L.remove(vehicles_waiting_4_L[0])
								if vehicles_waiting_4_L[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_4_L[0])
								if vehicles_waiting_4_L[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_4_L[0])
								vehicles_waiting_4_L.pop(0)
								position4_L.pop(0)
							else:
								position4_L = change_position4_L


					traci.simulationStep()


					add_vehicles_1_S = traci.inductionloop.getLastStepVehicleIDs("e1Detector_1i_1_1")
					add_vehicles_1_L = traci.inductionloop.getLastStepVehicleIDs("e1Detector_1i_2_1")

					remove_vehicles_1_S = traci.inductionloop.getLastStepVehicleIDs("e1Detector_3o_1_7")
					remove_vehicles_1_L = traci.inductionloop.getLastStepVehicleIDs("e1Detector_4o_2_8")


					add_vehicles_2_S = traci.inductionloop.getLastStepVehicleIDs("e1Detector_2i_1_2")
					add_vehicles_2_L = traci.inductionloop.getLastStepVehicleIDs("e1Detector_2i_2_2")

					remove_vehicles_2_S = traci.inductionloop.getLastStepVehicleIDs("e1Detector_4o_1_8")
					remove_vehicles_2_L = traci.inductionloop.getLastStepVehicleIDs("e1Detector_1o_2_5")


					add_vehicles_3_S = traci.inductionloop.getLastStepVehicleIDs("e1Detector_3i_1_3")
					add_vehicles_3_L = traci.inductionloop.getLastStepVehicleIDs("e1Detector_3i_2_3")

					remove_vehicles_3_S = traci.inductionloop.getLastStepVehicleIDs("e1Detector_1o_1_5")
					remove_vehicles_3_L = traci.inductionloop.getLastStepVehicleIDs("e1Detector_2o_2_6")


					add_vehicles_4_S = traci.inductionloop.getLastStepVehicleIDs("e1Detector_4i_1_4")
					add_vehicles_4_L = traci.inductionloop.getLastStepVehicleIDs("e1Detector_4i_2_4")

					remove_vehicles_4_S = traci.inductionloop.getLastStepVehicleIDs("e1Detector_2o_1_6")
					remove_vehicles_4_L = traci.inductionloop.getLastStepVehicleIDs("e1Detector_3o_2_7")


					add_vehicles(add_vehicles_1_L)
					add_vehicles(add_vehicles_1_S)

					add_vehicles(add_vehicles_2_L)
					add_vehicles(add_vehicles_2_S)

					add_vehicles(add_vehicles_3_L)
					add_vehicles(add_vehicles_3_S)

					add_vehicles(add_vehicles_4_L)
					add_vehicles(add_vehicles_4_S)

					remove_vehicles(remove_vehicles_1_L)
					remove_vehicles(remove_vehicles_1_S)

					remove_vehicles(remove_vehicles_2_L)
					remove_vehicles(remove_vehicles_2_S)

					remove_vehicles(remove_vehicles_3_L)
					remove_vehicles(remove_vehicles_3_S)

					remove_vehicles(remove_vehicles_4_L)
					remove_vehicles(remove_vehicles_4_S)

					for vehicle1_S in add_vehicles_1_S:
						if vehicle1_S not in vehicles_in_loop_1_S:
							vehicles_in_loop_1_S.append(vehicle1_S)
						if vehicle1_S not in vehicles_waiting_1_S:
							vehicles_waiting_1_S.append(vehicle1_S)
							if position1_S:
								i1 = len(position1_S)
								tmp_pos1_S = position1_S[i1 - 1]
								position1_S.append(tmp_pos1_S - 3)
								traci.vehicle.setStop(vehicle1_S, "1i", position1_S[i1], 1)
							else:
								for i1 in range(0, len(vehicles_waiting_1_S)):
									position1_S.append(485 - 3 * i1)
									traci.vehicle.setStop(vehicle1_S, "1i", position1_S[i1], 1)


					for vehicle1_L in add_vehicles_1_L:
						if vehicle1_L not in vehicles_in_loop_1_L:
							vehicles_in_loop_1_L.append(vehicle1_L)
						if vehicle1_L not in vehicles_waiting_1_L:
							vehicles_waiting_1_L.append(vehicle1_L)
							if position1_L:
								i3 = len(position1_L)
								tmp_pos1_L = position1_L[i3 - 1]
								position1_L.append(tmp_pos1_L - 3)
								traci.vehicle.setStop(vehicle1_L, "1i", position1_L[i3], 2)
							else:
								for i3 in range(0, len(vehicles_waiting_1_L)):
									position1_L.append(485 - 3 * i3)
									traci.vehicle.setStop(vehicle1_L, "1i", position1_L[i3], 2)


					for vehicle2_S in add_vehicles_2_S:
						if vehicle2_S not in vehicles_in_loop_2_S:
							vehicles_in_loop_2_S.append(vehicle2_S)
						if vehicle2_S not in vehicles_waiting_2_S:
							vehicles_waiting_2_S.append(vehicle2_S)
							if position2_S:
								j1 = len(position2_S)
								tmp_pos2_S = position2_S[j1 - 1]
								position2_S.append(tmp_pos2_S - 3)
								traci.vehicle.setStop(vehicle2_S, "2i", position2_S[j1], 1)
							else:
								for j1 in range(0, len(vehicles_waiting_2_S)):
									position2_S.append(485 - 3 * j1)
									traci.vehicle.setStop(vehicle2_S, "2i", position2_S[j1], 1)


					for vehicle2_L in add_vehicles_2_L:
						if vehicle2_L not in vehicles_in_loop_2_L:
							vehicles_in_loop_2_L.append(vehicle2_L)
						if vehicle2_L not in vehicles_waiting_2_L:
							vehicles_waiting_2_L.append(vehicle2_L)
							if position2_L:
								j3 = len(position2_L)
								tmp_pos2_L = position2_L[j3 - 1]
								position2_L.append(tmp_pos2_L - 3)
								traci.vehicle.setStop(vehicle2_L, "2i", position2_L[j3], 2)
							else:
								for j3 in range(0, len(vehicles_waiting_2_L)):
									position2_L.append(485 - 3 * j3)
									traci.vehicle.setStop(vehicle2_L, "2i", position2_L[j3], 2)


					for vehicle3_S in add_vehicles_3_S:
						if vehicle3_S not in vehicles_in_loop_3_S:
							vehicles_in_loop_3_S.append(vehicle3_S)
						if vehicle3_S not in vehicles_waiting_3_S:
							vehicles_waiting_3_S.append(vehicle3_S)
							if position3_S:
								k1 = len(position3_S)
								tmp_pos3_S = position3_S[k1 - 1]
								position3_S.append(tmp_pos3_S - 3)
								traci.vehicle.setStop(vehicle3_S, "3i", position3_S[k1], 1)
							else:
								for k1 in range(0, len(vehicles_waiting_3_S)):
									position3_S.append(485 - 3 * k1)
									traci.vehicle.setStop(vehicle3_S, "3i", position3_S[k1], 1)


					for vehicle3_L in add_vehicles_3_L:
						if vehicle3_L not in vehicles_in_loop_3_L:
							vehicles_in_loop_3_L.append(vehicle3_L)
						if vehicle3_L not in vehicles_waiting_3_L:
							vehicles_waiting_3_L.append(vehicle3_L)
							if position3_L:
								k3 = len(position3_L)
								tmp_pos3_L = position3_L[k3 - 1]
								position3_L.append(tmp_pos3_L - 3)
								traci.vehicle.setStop(vehicle3_L, "3i", position3_L[k3], 2)
							else:
								for k3 in range(0, len(vehicles_waiting_3_L)):
									position3_L.append(485 - 3 * k3)
									traci.vehicle.setStop(vehicle3_L, "3i", position3_L[k3], 2)


					for vehicle4_S in add_vehicles_4_S:
						if vehicle4_S not in vehicles_in_loop_4_S:
							vehicles_in_loop_4_S.append(vehicle4_S)
						if vehicle4_S not in vehicles_waiting_4_S:
							vehicles_waiting_4_S.append(vehicle4_S)
							if position4_S:
								l1 = len(position4_S)
								tmp_pos4_S = position4_S[l1 - 1]
								position4_S.append(tmp_pos4_S - 3)
								traci.vehicle.setStop(vehicle4_S, "4i", position4_S[l1], 1)
							else:
								for l1 in range(0, len(vehicles_waiting_4_S)):
									position4_S.append(485 - 3 * l1)
									traci.vehicle.setStop(vehicle4_S, "4i", position4_S[l1], 1)


					for vehicle4_L in add_vehicles_4_L:
						if vehicle4_L not in vehicles_in_loop_4_L:
							vehicles_in_loop_4_L.append(vehicle4_L)
						if vehicle4_L not in vehicles_waiting_4_L:
							vehicles_waiting_4_L.append(vehicle4_L)
							if position4_L:
								l3 = len(position4_L)
								tmp_pos4_L = position4_L[l3 - 1]
								position4_L.append(tmp_pos4_L - 3)
								traci.vehicle.setStop(vehicle4_L, "4i", position4_L[l3], 2)
							else:
								for l3 in range(0, len(vehicles_waiting_4_L)):
									position4_L.append(485 - 3 * l3)
									traci.vehicle.setStop(vehicle4_L, "4i", position4_L[l3], 2)

					if remove_vehicles_1_S:
						if remove_vehicles_1_S[0] in vehicles_in_loop_1_S:
							vehicles_in_loop_1_S.pop(0)

					if remove_vehicles_1_L:
						if remove_vehicles_1_L[0] in vehicles_in_loop_1_L:
							vehicles_in_loop_1_L.pop(0)

					if remove_vehicles_2_S:
						if remove_vehicles_2_S[0] in vehicles_in_loop_2_S:
							vehicles_in_loop_2_S.pop(0)

					if remove_vehicles_2_L:
						if remove_vehicles_2_L[0] in vehicles_in_loop_2_L:
							vehicles_in_loop_2_L.pop(0)

					if remove_vehicles_3_S:
						if remove_vehicles_3_S[0] in vehicles_in_loop_3_S:
							vehicles_in_loop_3_S.pop(0)

					if remove_vehicles_3_L:
						if remove_vehicles_3_L[0] in vehicles_in_loop_3_L:
							vehicles_in_loop_3_L.pop(0)

					if remove_vehicles_4_S:
						if remove_vehicles_4_S[0] in vehicles_in_loop_4_S:
							vehicles_in_loop_4_S.pop(0)

					if remove_vehicles_4_L:
						if remove_vehicles_4_L[0] in vehicles_in_loop_4_L:
							vehicles_in_loop_4_L.pop(0)


					S_ = set_state_space(S_, vehicles_in_loop_1_L, vehicles_in_loop_1_S, vehicles_in_loop_2_L,
									vehicles_in_loop_2_S, vehicles_in_loop_3_L, vehicles_in_loop_3_S,
									vehicles_in_loop_4_L, vehicles_in_loop_4_S)
									
					R = len(remove_vehicles_1S) + len(remove_vehicles_2S) + len(remove_vehicles_3S) + len(remove_vehicles_4S) + len(remove_vehicles_1L) 
					+ len(remove_vehicles_2L) + len(remove_vehicles_3L) + len(remove_vehicles_4L)

					if remove_vehicles_1L:
						A = 'Lane1L'
						B[0] = 0

					if remove_vehicles_1S:
						A = 'Lane1S'
						B[0] = 1

					if remove_vehicles_2L:
						A = 'Lane2L'
						B[0] = 2

					if remove_vehicles_2S:
						A = 'Lane2S'
						B[0] = 3
						
					if remove_vehicles_3L:
						A = 'Lane3L'
						B[0] = 0

					if remove_vehicles_3S:
						A = 'Lane3S'
						B[0] = 1

					if remove_vehicles_4L:
						A = 'Lane4L'
						B[0] = 2

					if remove_vehicles_4S:
						A = 'Lane4S'
						B[0] = 3

					R_ = traci.simulation.getCollidingVehiclesNumber()
					R_List = traci.simulation.getCollidingVehiclesIDList()

					for vehicleC in R_List:
						if vehicleC not in collided_vehicles:
							collided_vehicles.append(vehicleC)
							list_is_new = True

					if R_ and list_is_new == True:

						R_collision = 1
						collision_reward = -1
						R_ = R_ * collision_reward / 2.5

						c = len(collided_vehicles)
						d1 = collided_vehicles[c - 1]
						d2 = collided_vehicles[c - 2]

						m = ['rightU', 'right', 'upL', 'up', 'leftD', 'left', 'downL', 'down']

						pos1 = 0
						pos2 = 0
						n1 = 0
						n2 = 0
						col_action = 0

						for n in range(len(m)):
							m[n] = re.compile(m[n])
							p = m[n].match(d1)
							if p:
								if n == 0:
									pos1 = S_old[0, 0]
									n1 = 0
									break

								if n == 1:
									pos1 = S_old[0, 1]
									n1 = 1
									break

								if n == 2:
									pos1 = S_old[0, 2]
									n1 = 2
									break

								if n == 3:
									pos1 = S_old[0, 3]
									n1 = 3
									break

								if n == 4:
									pos1 = S_old[0, 4]
									n1 = 4
									break

								if n == 5:
									pos1 = S_old[0, 5]
									n1 = 5
									break

								if n == 6:
									pos1 = S_old[0, 6]
									n1 = 6
									break

								if n == 7:
									pos1 = S_old[0, 7]
									n1 = 7
									break

						for n in range(len(m)):
							m[n] = re.compile(m[n])
							p = m[n].match(d2)
							if p:
								if n == 0:
									pos2 = S_old[0, 0]
									n2 = 0
									break

								if n == 1:
									pos2 = S_old[0, 1]
									n2 = 1
									break

								if n == 2:
									pos2 = S_old[0, 2]
									n2 = 2
									break

								if n == 3:
									pos2 = S_old[0, 3]
									n2 = 3
									break

								if n == 4:
									pos2 = S_old[0, 4]
									n2 = 4
									break

								if n == 5:
									pos2 = S_old[0, 5]
									n2 = 5
									break

								if n == 6:
									pos2 = S_old[0, 6]
									n2 = 6
									break

								if n == 7:
									pos2 = S_old[0, 7]
									n2 = 7
									break


						if pos1 > 440 and pos2 > 440:
							if pos1 > pos2:
								col_action = n2
							else:
								col_action = n1
						if pos1 > 440 and pos2 < 440:
							col_action = n1
						if pos1 < 440 and pos2 > 440:
							col_action = n2
						if pos1 < 440 and pos2 < 440:
							if pos1 > pos2:
								col_action = n2
							else:
								col_action = n1

						if col_action == 0:
							A_old = 'Lane1L'
							B_old[0] = 0

						if col_action == 1:
							A_old = 'Lane1S'
							B_old[0] = 1

						if col_action == 2:
							A_old = 'Lane2L'
							B_old[0] = 2

						if col_action == 3:
							A_old = 'Lane2S'
							B_old[0] = 3

						if col_action == 4:
							A_old = 'Lane3L'
							B_old[0] = 4

						if col_action == 5:
							A_old = 'Lane3S'
							B_old[0] = 5

						if col_action == 6:
							A_old = 'Lane4L'
							B_old[0] = 6

						if col_action == 7:
							A_old = 'Lane4S'
							B_old[0] = 7


						allQ = sess.run(Qout, feed_dict={input_NN:S_old})
						Q1 = sess.run(Qout, feed_dict={input_NN:S_})

						maxQ1 = np.max(Q1)
						targetQ = allQ

						targetQ[0, B_old[0]] = R_ + GAMMA * maxQ1

						_, W1, W2 = sess.run([updateModel, weights1, weights2], feed_dict={input_NN:S_old, nextQ:targetQ})

						episodeBuffer.add(np.reshape(np.array([S_old, B_old[0], R_, S_]), [1, 4]))

					elif R:

						Q1 = sess.run(Qout, feed_dict={input_NN:S_})

						maxQ1 = np.max(Q1)
						targetQ = allQ

						targetQ[0, B[0]] = R + GAMMA * maxQ1

						_, W1, W2 = sess.run([updateModel, weights1, weights2], feed_dict={input_NN:S, nextQ:targetQ})

						episodeBuffer.add(np.reshape(np.array([S, B[0], R, S_]), [1, 4]))

					total_reward += R + R_
					step += 1
					global_step += 1

					EPSILON = min(EPSILON + (1 - EPSILON_Orig) / (MAX_EXPLORATION_STEPS), 0.99)

					if global_step > PRE_TRAIN_STEPS:
						if step % (update_frequency) == 0:
							trainBatch = myBuffer.sample(batch_size)

							for train in range(0, len(trainBatch)):

								allQ = sess.run(Qout, feed_dict={input_NN:trainBatch[train, 0]})
								Q1 = sess.run(Qout, feed_dict={input_NN:trainBatch[train, 3]})
								maxQ1 = np.max(Q1)
								targetQ = allQ

								targetQ[0, trainBatch[train, 1]] = trainBatch[train, 2] + GAMMA * maxQ1
								_, W1, W2 = sess.run([updateModel, weights1, weights2], feed_dict={input_NN:trainBatch[train, 0], nextQ:targetQ})

					myBuffer.add(episodeBuffer.buffer)

					S = S_

					for i in range (0, len(vehicles_in_loop_1_L)):
						co2_1L = 0
						co2_1L = traci.vehicle.getCO2Emission(vehicles_in_loop_1_L[i])
						Tot_co2_1L += co2_1L

					if vehicles_waiting_1_L:
						for j in range(0, len(vehicles_waiting_1_L)):
							wt_1L = traci.vehicle.getWaitingTime(vehicles_waiting_1_L[j])
							Tot_wt_time += wt_1L

					for i in range (0, len(vehicles_in_loop_1_S)):
						co2_1S = 0
						co2_1S = traci.vehicle.getCO2Emission(vehicles_in_loop_1_S[i])
						Tot_co2_1S += co2_1S

					if vehicles_waiting_1_S:
						for j in range(0, len(vehicles_waiting_1_S)):
							wt_1S = traci.vehicle.getWaitingTime(vehicles_waiting_1_S[j])
							Tot_wt_time += wt_1S

					for i in range(0, len(vehicles_in_loop_2_L)):
						co2_2L = 0
						co2_2L = traci.vehicle.getCO2Emission(vehicles_in_loop_2_L[i])
						Tot_co2_2L += co2_2L

					if vehicles_waiting_2_L:
						for j in range(0, len(vehicles_waiting_2_L)):
							wt_2L = traci.vehicle.getWaitingTime(vehicles_waiting_2_L[j])
							Tot_wt_time += wt_2L

					for i in range(0, len(vehicles_in_loop_2_S)):
						co2_2S = 0
						co2_2S = traci.vehicle.getCO2Emission(vehicles_in_loop_2_S[i])
						Tot_co2_2S += co2_2S

					if vehicles_waiting_2_S:
						for j in range(0, len(vehicles_waiting_2_S)):
							wt_2S = traci.vehicle.getWaitingTime(vehicles_waiting_2_S[j])
							Tot_wt_time += wt_2S

					for i in range(0, len(vehicles_in_loop_3_L)):
						co2_3L = 0
						co2_3L = traci.vehicle.getCO2Emission(vehicles_in_loop_3_L[i])
						Tot_co2_3L += co2_3L

					if vehicles_waiting_3_L:
						for j in range(0, len(vehicles_waiting_3_L)):
							wt_3L = traci.vehicle.getWaitingTime(vehicles_waiting_3_L[j])
							Tot_wt_time += wt_3L

					for i in range(0, len(vehicles_in_loop_3_S)):
						co2_3S = 0
						co2_3S = traci.vehicle.getCO2Emission(vehicles_in_loop_3_S[i])
						Tot_co2_3S += co2_3S

					if vehicles_waiting_3_S:
						for j in range(0, len(vehicles_waiting_3_S)):
							wt_3S = traci.vehicle.getWaitingTime(vehicles_waiting_3_S[j])
							Tot_wt_time += wt_3S

					for i in range(0, len(vehicles_in_loop_4_S)):
						co2_4S = 0
						co2_4S = traci.vehicle.getCO2Emission(vehicles_in_loop_4_S[i])
						Tot_co2_4S += co2_4S

					if vehicles_waiting_4_S:
						for j in range(0, len(vehicles_waiting_4_S)):
							wt_4S = traci.vehicle.getWaitingTime(vehicles_waiting_4_S[j])
							Tot_wt_time += wt_4S

					for i in range(0, len(vehicles_in_loop_4_L)):
						co2_4L = 0
						co2_4L = traci.vehicle.getCO2Emission(vehicles_in_loop_4_L[i])
						Tot_co2_4L += co2_4L

					if vehicles_waiting_4_L:
						for j in range(0, len(vehicles_waiting_4_L)):
							wt_4L = traci.vehicle.getWaitingTime(vehicles_waiting_4_L[j])
							Tot_wt_time += wt_4L


					total_co2_emission += Tot_co2_1L + Tot_co2_1S + Tot_co2_2L + Tot_co2_2S + Tot_co2_3L + Tot_co2_3S + Tot_co2_4S + Tot_co2_4L
					waiting_time += wt_time_1L + wt_time_1S + wt_time_2L + wt_time_2S + wt_time_3L + wt_time_3S + wt_time_4L + wt_time_4S
					
								
				if test:
				
					R_ = 0
					R = 0
					R_collision = 0
					wt_time_1L = 0
					wt_time_1S = 0
					wt_time_2L = 0
					wt_time_2S = 0
					wt_time_3L = 0
					wt_time_3S = 0
					wt_time_4L = 0
					wt_time_4S = 0

					wt_1L = 0
					wt_1S = 0
					wt_2L = 0
					wt_2S = 0
					wt_3L = 0
					wt_3S = 0
					wt_4L = 0
					wt_4S = 0

					Tot_co2_1L = 0
					Tot_co2_1S = 0
					Tot_co2_2L = 0
					Tot_co2_2S = 0
					Tot_co2_3L = 0
					Tot_co2_3S = 0
					Tot_co2_4L = 0
					Tot_co2_4S = 0
					list_is_new = False

					if len(running_vehicle_ids) > 300:
						running_vehicle_ids = []
					if len(lane_change_disabled_vehicles) > 300:
						lane_change_disabled_vehicles = []

					running_vehicle_ids = traci.vehicle.getIDList()
					for vehicleID in running_vehicle_ids:
						if vehicleID not in lane_change_disabled_vehicles:
							traci.vehicle.setSpeedMode(vehicleID, 0b00000)
							traci.vehicle.setLaneChangeMode(vehicleID, 0b0100000000)
							lane_change_disabled_vehicles.append(vehicleID)

					S_old = S.copy()
					S_col = S_old.copy()
					A_old = A
					B_old = B.copy()

					B, allQ = sess.run([predict, Qout], feed_dict={input_NN:S})

					S_ = S.copy()

					if B == 0:
						A = 'Lane1L'
					if B == 1:
						A = 'Lane1S'
					if B == 2:
						A = 'Lane2L'
					if B == 3:
						A = 'Lane2S'
					if B == 4:
						A = 'Lane3L'
					if B == 5:
						A = 'Lane3S'
					if B == 6:
						A = 'Lane4L'
					if B == 7:
						A = 'Lane4S'

					if np.random.uniform() > EPSILON:
						if (vehicles_waiting_1_S or vehicles_waiting_1_L or
								vehicles_waiting_2_L or vehicles_waiting_2_S or
								vehicles_waiting_3_S or vehicles_waiting_3_L or
								vehicles_waiting_4_S or vehicles_waiting_4_L):
							while(1):
								ran_int = np.random.randint(0, 8)

								if ran_int == 0:
									if vehicles_waiting_1_L:
										A = 'Lane1L'
										break

								if ran_int == 1:
									if vehicles_waiting_1_S:
										A = 'Lane1S'
										break

								if ran_int == 2:
									if vehicles_waiting_2_L:
										A = 'Lane2L'
										break

								if ran_int == 3:
									if vehicles_waiting_2_S:
										A = 'Lane2S'
										break

								if ran_int == 4:
									if vehicles_waiting_3_L:
										A = 'Lane3L'
										break

								if ran_int == 5:
									if vehicles_waiting_3_S:
										A = 'Lane3S'
										break

								if ran_int == 6:
									if vehicles_waiting_4_L:
										A = 'Lane4L'
										break

								if ran_int == 7:
									if vehicles_waiting_4_S:
										A = 'Lane4S'
										break
						else:
							A = np.random.choice(ACTIONS)


					if (A == 'Lane1L' or ((S[0, 0] > 0) and np.equal(S[0, 1], 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
										  and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))
							or ((S[0, 0] > 0) and (S[0, 1] > 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
								and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))):

						if vehicles_waiting_1_L:
							try:
								traci.vehicle.setStop(vehicles_waiting_1_L[0], "1i", position1_L[0], 2, 0)
								traci.vehicle.setSpeed(vehicles_waiting_1_L[0], 10)
								traci.vehicle.setSpeedMode(vehicles_waiting_1_L[0], 0)
								wt_time_1L = traci.vehicle.getWaitingTime(vehicles_waiting_1_L[0])
								vehicles_waiting_1_L.pop(0)
								position1_L.pop(0)
								A = 'Lane1L'
								B[0] = 0
							except Exception:
								traci.vehicle.remove(vehicles_waiting_1_L[0], reason=3)
								if vehicles_waiting_1_L[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_1_L[0])
								vehicles_in_loop_1_L.remove(vehicles_waiting_1_L[0])
								if vehicles_waiting_1_L[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_1_L[0])
								if vehicles_waiting_1_L[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_1_L[0])
								vehicles_waiting_1_L.pop(0)
								position1_L.pop(0)


					if (A == 'Lane1S' or (np.equal(S[0, 0], 0) and (S[0, 1] > 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
										  and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))
							or ((S[0, 0] > 0) and (S[0, 1] > 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
								and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))):

						if vehicles_waiting_1_S:
							try:
								traci.vehicle.setStop(vehicles_waiting_1_S[0], "1i", position1_S[0], 1, 0)
								traci.vehicle.setSpeed(vehicles_waiting_1_S[0], 10)
								traci.vehicle.setSpeedMode(vehicles_waiting_1_S[0], 0)
								wt_time_1S = traci.vehicle.getWaitingTime(vehicles_waiting_1_S[0])
								vehicles_waiting_1_S.pop(0)
								position1_S.pop(0)
								A = 'Lane1S'
								B[0] = 1
							except Exception:
								traci.vehicle.remove(vehicles_waiting_1_S[0], reason=3)
								if vehicles_waiting_1_S[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_1_S[0])
								vehicles_in_loop_1_S.remove(vehicles_waiting_1_S[0])
								if vehicles_waiting_1_S[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_1_S[0])
								if vehicles_waiting_1_S[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_1_S[0])
								vehicles_waiting_1_S.pop(0)
								position1_S.pop(0)


					if (A == 'Lane2L' or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and (S[0, 2] > 0) and np.equal(S[0, 3], 0)
										  and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))
							or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and (S[0, 2] > 0) and (S[0, 3] > 0)
								and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))):

						if vehicles_waiting_2_L:
							try:
								traci.vehicle.setStop(vehicles_waiting_2_L[0], "2i", position2_L[0], 2, 0)
								traci.vehicle.setSpeed(vehicles_waiting_2_L[0], 10)
								traci.vehicle.setSpeedMode(vehicles_waiting_2_L[0], 0)
								wt_time_2L = traci.vehicle.getWaitingTime(vehicles_waiting_2_L[0])
								vehicles_waiting_2_L.pop(0)
								position2_L.pop(0)
								A = 'Lane2L'
								B[0] = 2
							except Exception:
								traci.vehicle.remove(vehicles_waiting_2_L[0], reason=3)
								if vehicles_waiting_2_L[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_2_L[0])
								vehicles_in_loop_2_L.remove(vehicles_waiting_2_L[0])
								if vehicles_waiting_2_L[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_2_L[0])
								if vehicles_waiting_2_L[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_2_L[0])
								vehicles_waiting_2_L.pop(0)
								position2_L.pop(0)


					if (A == 'Lane2S' or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and np.equal(S[0, 2], 0) and (S[0, 3] > 0)
										  and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))
							or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and (S[0, 2] > 0) and (S[0, 3] > 0)
								and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))):

						if vehicles_waiting_2_S:
							try:
								traci.vehicle.setStop(vehicles_waiting_2_S[0], "2i", position2_S[0], 1, 0)
								traci.vehicle.setSpeed(vehicles_waiting_2_S[0], 10)
								traci.vehicle.setSpeedMode(vehicles_waiting_2_S[0], 0)
								wt_time_2S = traci.vehicle.getWaitingTime(vehicles_waiting_2_S[0])
								vehicles_waiting_2_S.pop(0)
								position2_S.pop(0)
								A = 'Lane2S'
								B[0] = 3
							except Exception:
								traci.vehicle.remove(vehicles_waiting_2_S[0], reason=3)
								if vehicles_waiting_2_S[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_2_S[0])
								vehicles_in_loop_2_S.remove(vehicles_waiting_2_S[0])
								if vehicles_waiting_2_S[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_2_S[0])
								if vehicles_waiting_2_S[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_2_S[0])
								vehicles_waiting_2_S.pop(0)
								position2_S.pop(0)


					if (A == 'Lane3L' or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
										  and (S[0, 4] > 0) and np.equal(S[0, 5], 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))
							or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
								and (S[0, 4] > 0) and (S[0, 5] > 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))):

						if vehicles_waiting_3_L:
							try:
								traci.vehicle.setStop(vehicles_waiting_3_L[0], "3i", position3_L[0], 2, 0)
								traci.vehicle.setSpeed(vehicles_waiting_3_L[0], 10)
								traci.vehicle.setSpeedMode(vehicles_waiting_3_L[0], 0)
								wt_time_3L = traci.vehicle.getWaitingTime(vehicles_waiting_3_L[0])
								vehicles_waiting_3_L.pop(0)
								position3_L.pop(0)
								A = 'Lane3L'
								B[0] = 4
							except Exception:
								traci.vehicle.remove(vehicles_waiting_3_L[0], reason=3)
								if vehicles_waiting_3_L[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_3_L[0])
								vehicles_in_loop_3_L.remove(vehicles_waiting_3_L[0])
								if vehicles_waiting_3_L[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_3_L[0])
								if vehicles_waiting_3_L[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_3_L[0])
								vehicles_waiting_3_L.pop(0)
								position3_L.pop(0)

					if (A == 'Lane3S' or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
										  and np.equal(S[0, 4], 0) and (S[0, 5] > 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))
							or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
								and (S[0, 4] > 0) and (S[0, 5] > 0) and np.equal(S[0, 6], 0) and np.equal(S[0, 7], 0))):

						if vehicles_waiting_3_S:
							try:
								traci.vehicle.setStop(vehicles_waiting_3_S[0], "3i", position3_S[0], 1, 0)
								traci.vehicle.setSpeed(vehicles_waiting_3_S[0], 10)
								traci.vehicle.setSpeedMode(vehicles_waiting_3_S[0], 0)
								wt_time_3S = traci.vehicle.getWaitingTime(vehicles_waiting_3_S[0])
								vehicles_waiting_3_S.pop(0)
								position3_S.pop(0)
								A = 'Lane3S'
								B[0] = 5
							except:
								traci.vehicle.remove(vehicles_waiting_3_S[0], reason=3)
								if vehicles_waiting_3_S[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_3_S[0])
								vehicles_in_loop_3_S.remove(vehicles_waiting_3_S[0])
								if vehicles_waiting_3_S[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_3_S[0])
								if vehicles_waiting_3_S[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_3_S[0])
								vehicles_waiting_3_S.pop(0)
								position3_S.pop(0)


					if (A == 'Lane4L' or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
										  and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and (S[0, 6] > 0) and np.equal(S[0, 7], 0))
							or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
								and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and (S[0, 6] > 0) and (S[0, 7] > 0))):

						if vehicles_waiting_4_L:
							try:
								traci.vehicle.setStop(vehicles_waiting_4_L[0], "4i", position4_L[0], 2, 0)
								traci.vehicle.setSpeed(vehicles_waiting_4_L[0], 10)
								traci.vehicle.setSpeedMode(vehicles_waiting_4_L[0], 0)
								wt_time_4L = traci.vehicle.getWaitingTime(vehicles_waiting_4_L[0])
								vehicles_waiting_4_L.pop(0)
								position4_L.pop(0)
								A = 'Lane4L'
								B[0] = 6
							except Exception:
								traci.vehicle.remove(vehicles_waiting_4_L[0], reason=3)
								if vehicles_waiting_4_L[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_4_L[0])
								vehicles_in_loop_4_L.remove(vehicles_waiting_4_L[0])
								if vehicles_waiting_4_L[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_4_L[0])
								if vehicles_waiting_4_L[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_4_L[0])
								vehicles_waiting_4_L.pop(0)
								position4_L.pop(0)


					if (A == 'Lane4S' or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
										  and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and np.equal(S[0, 6], 0) and (S[0, 7] > 0))
							or (np.equal(S[0, 0], 0) and np.equal(S[0, 1], 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 3], 0)
								and np.equal(S[0, 4], 0) and np.equal(S[0, 5], 0) and (S[0, 6] > 0) and (S[0, 7] > 0))):

						if vehicles_waiting_4_S:
							try:
								traci.vehicle.setStop(vehicles_waiting_4_S[0], "4i", position4_S[0], 1, 0)
								traci.vehicle.setSpeed(vehicles_waiting_4_S[0], 10)
								traci.vehicle.setSpeedMode(vehicles_waiting_4_S[0], 0)
								wt_time_4S = traci.vehicle.getWaitingTime(vehicles_waiting_4_S[0])
								vehicles_waiting_4_S.pop(0)
								position4_S.pop(0)
								A = 'Lane4S'
								B[0] = 7
							except Exception:
								traci.vehicle.remove(vehicles_waiting_4_S[0], reason=3)
								if vehicles_waiting_4_S[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_4_S[0])
								vehicles_in_loop_4_S.remove(vehicles_waiting_4_S[0])
								if vehicles_waiting_4_S[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_4_S[0])
								if vehicles_waiting_4_S[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_4_S[0])
								vehicles_waiting_4_S.pop(0)
								position4_S.pop(0)


					if 485 not in position1_S:
						if vehicles_waiting_1_S:
							orig_position1_S = position1_S.copy()
							pos1_S = 485
							i1S = 0
							except1_S = 0
							change_position1_S = []
							for vehicle1_S in vehicles_waiting_1_S:
								try:
									traci.vehicle.setStop(vehicle1_S, "1i", position1_S[i1S], 1, 0)
									traci.vehicle.setStop(vehicle1_S, "1i", pos1_S, 1)
									change_position1_S.append(pos1_S)
									pos1_S -= 3
									i1S += 1
								except Exception:
									except1_S = 1
									continue
							if except1_S:
								traci.vehicle.remove(vehicles_waiting_1_S[0], reason=3)
								if vehicles_waiting_1_S[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_1_S[0])
								vehicles_in_loop_1_S.remove(vehicles_waiting_1_S[0])
								if vehicles_waiting_1_S[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_1_S[0])
								if vehicles_waiting_1_S[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_1_S[0])
								vehicles_waiting_1_S.pop(0)
								position1_S.pop(0)
							else:
								position1_S = change_position1_S

					if 485 not in position1_L:
						if vehicles_waiting_1_L:
							pos1_L = 485
							i1L = 0
							orig_position1_L = position1_L.copy()
							except1_L = 0
							change_position1_L = []
							for vehicle1_L in vehicles_waiting_1_L:
								try:
									traci.vehicle.setStop(vehicle1_L, "1i", position1_L[i1L], 2, 0)
									traci.vehicle.setStop(vehicle1_L, "1i", pos1_L, 2)
									change_position1_L.append(pos1_L)
									pos1_L -= 3
									i1L += 1
								except Exception:
									except1_L = 1
									continue
							if except1_L:
								traci.vehicle.remove(vehicles_waiting_1_L[0], reason=3)
								if vehicles_waiting_1_L[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_1_L[0])
								vehicles_in_loop_1_L.remove(vehicles_waiting_1_L[0])
								if vehicles_waiting_1_L[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_1_L[0])
								if vehicles_waiting_1_L[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_1_L[0])
								vehicles_waiting_1_L.pop(0)
								position1_L.pop(0)
							else:
								position1_L = change_position1_L

					if 485 not in position2_S:
						if vehicles_waiting_2_S:
							pos2_S = 485
							i2S = 0
							except2_S = 0
							orig_position2_S = position2_S.copy()
							change_position2_S = []
							for vehicle2_S in vehicles_waiting_2_S:
								try:
									traci.vehicle.setStop(vehicle2_S, "2i", position2_S[i2S], 1, 0)
									traci.vehicle.setStop(vehicle2_S, "2i", pos2_S, 1)
									change_position2_S.append(pos2_S)
									pos2_S -= 3
									i2S += 1
								except Exception:
									except2_S = 1
									continue
							if except2_S:
								traci.vehicle.remove(vehicles_waiting_2_S[0], reason=3)
								if vehicles_waiting_2_S[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_2_S[0])
								vehicles_in_loop_2_S.remove(vehicles_waiting_2_S[0])
								if vehicles_waiting_2_S[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_2_S[0])
								if vehicles_waiting_2_S[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_2_S[0])
								vehicles_waiting_2_S.pop(0)
								position2_S.pop(0)
							else:
								position2_S = change_position2_S

					if 485 not in position2_L:
						if vehicles_waiting_2_L:
							pos2_L = 485
							i2L = 0
							orig_position2_L = position2_L.copy()
							except2_L = 0
							change_position2_L = []
							for vehicle2_L in vehicles_waiting_2_L:
								try:
									traci.vehicle.setStop(vehicle2_L, "2i", position2_L[i2L], 2, 0)
									traci.vehicle.setStop(vehicle2_L, "2i", pos2_L, 2)
									change_position2_L.append(pos2_L)
									pos2_L -= 3
									i2L += 1
								except Exception:
									except2_L = 1
									continue
							if except2_L:
								traci.vehicle.remove(vehicles_waiting_2_L[0], reason=3)
								if vehicles_waiting_2_L[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_2_L[0])
								vehicles_in_loop_2_L.remove(vehicles_waiting_2_L[0])
								if vehicles_waiting_2_L[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_2_L[0])
								if vehicles_waiting_2_L[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_2_L[0])
								vehicles_waiting_2_L.pop(0)
								position2_L.pop(0)
							else:
								position2_L = change_position2_L

					if 485 not in position3_S:
						if vehicles_waiting_3_S:
							pos3_S = 485
							i3S = 0
							orig_position3_S = position3_S.copy()
							except3_S = 0
							change_position3_S = []
							for vehicle3_S in vehicles_waiting_3_S:
								try:
									traci.vehicle.setStop(vehicle3_S, "3i", position3_S[i3S], 1, 0)
									traci.vehicle.setStop(vehicle3_S, "3i", pos3_S, 1)
									change_position3_S.append(pos3_S)
									pos3_S -= 3
									i3S += 1
								except Exception:
									except3_S = 1
									continue
							if except3_S:
								traci.vehicle.remove(vehicles_waiting_3_S[0], reason=3)
								if vehicles_waiting_3_S[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_3_S[0])
								vehicles_in_loop_3_S.remove(vehicles_waiting_3_S[0])
								if vehicles_waiting_3_S[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_3_S[0])
								if vehicles_waiting_3_S[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_3_S[0])
								vehicles_waiting_3_S.pop(0)
								position3_S.pop(0)
							else:
								position3_S = change_position3_S

					if 485 not in position3_L:
						if vehicles_waiting_3_L:
							pos3_L = 485
							i3L = 0
							orig_position3_L = position3_L.copy()
							except3_L = 0
							change_position3_L = []
							for vehicle3_L in vehicles_waiting_3_L:
								try:
									traci.vehicle.setStop(vehicle3_L, "3i", position3_L[i3L], 2, 0)
									traci.vehicle.setStop(vehicle3_L, "3i", pos3_L, 2)
									change_position3_L.append(pos3_L)
									pos3_L -= 3
									i3L += 1
								except Exception:
									except3_L = 1
									continue
							if except3_L:
								traci.vehicle.remove(vehicles_waiting_3_L[0], reason=3)
								if vehicles_waiting_3_L[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_3_L[0])
								vehicles_in_loop_3_L.remove(vehicles_waiting_3_L[0])
								if vehicles_waiting_3_L[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_3_L[0])
								if vehicles_waiting_3_L[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_3_L[0])
								vehicles_waiting_3_L.pop(0)
								position3_L.pop(0)
							else:
								position3_L = change_position3_L

					if 485 not in position4_S:
						if vehicles_waiting_4_S:
							pos4_S = 485
							i4S = 0
							except4_S = 0
							orig_position4_S = position4_S.copy()
							change_position4_S = []
							for vehicle4_S in vehicles_waiting_4_S:
								try:
									traci.vehicle.setStop(vehicle4_S, "4i", position4_S[i4S], 1, 0)
									traci.vehicle.setStop(vehicle4_S, "4i", pos4_S, 1)
									change_position4_S.append(pos4_S)
									pos4_S -= 3
									i4S += 1
								except Exception:
									except4_S = 1
									continue
							if except4_S:
								traci.vehicle.remove(vehicles_waiting_4_S[0], reason=3)
								if vehicles_waiting_4_S[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_4_S[0])
								vehicles_in_loop_4_S.remove(vehicles_waiting_4_S[0])
								if vehicles_waiting_4_S[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_4_S[0])
								if vehicles_waiting_4_S[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_4_S[0])
								vehicles_waiting_4_S.pop(0)
								position4_S.pop(0)
							else:
								position4_S = change_position4_S

					if 485 not in position4_L:
						if vehicles_waiting_4_L:
							pos4_L = 485
							i4L = 0
							orig_position4_L = position4_L.copy()
							except4_L = 0
							change_position4_L = []
							for vehicle4_L in vehicles_waiting_4_L:
								try:
									traci.vehicle.setStop(vehicle4_L, "4i", position4_L[i4L], 2, 0)
									traci.vehicle.setStop(vehicle4_L, "4i", pos4_L, 2)
									change_position4_L.append(pos4_L)
									pos4_L -= 3
									i4L += 1
								except Exception:
									except4_L = 1
									continue
							if except4_L:
								traci.vehicle.remove(vehicles_waiting_4_L[0], reason=3)
								if vehicles_waiting_4_L[0] in managed_vehicles:
									managed_vehicles.remove(vehicles_waiting_4_L[0])
								vehicles_in_loop_4_L.remove(vehicles_waiting_4_L[0])
								if vehicles_waiting_4_L[0] in running_vehicle_ids:
									running_vehicle_ids.remove(vehicles_waiting_4_L[0])
								if vehicles_waiting_4_L[0] in lane_change_disabled_vehicles:
									lane_change_disabled_vehicles.remove(vehicles_waiting_4_L[0])
								vehicles_waiting_4_L.pop(0)
								position4_L.pop(0)
							else:
								position4_L = change_position4_L


					traci.simulationStep()


					add_vehicles_1_S = traci.inductionloop.getLastStepVehicleIDs("e1Detector_1i_1_1")
					add_vehicles_1_L = traci.inductionloop.getLastStepVehicleIDs("e1Detector_1i_2_1")

					remove_vehicles_1_S = traci.inductionloop.getLastStepVehicleIDs("e1Detector_3o_1_7")
					remove_vehicles_1_L = traci.inductionloop.getLastStepVehicleIDs("e1Detector_4o_2_8")


					add_vehicles_2_S = traci.inductionloop.getLastStepVehicleIDs("e1Detector_2i_1_2")
					add_vehicles_2_L = traci.inductionloop.getLastStepVehicleIDs("e1Detector_2i_2_2")

					remove_vehicles_2_S = traci.inductionloop.getLastStepVehicleIDs("e1Detector_4o_1_8")
					remove_vehicles_2_L = traci.inductionloop.getLastStepVehicleIDs("e1Detector_1o_2_5")


					add_vehicles_3_S = traci.inductionloop.getLastStepVehicleIDs("e1Detector_3i_1_3")
					add_vehicles_3_L = traci.inductionloop.getLastStepVehicleIDs("e1Detector_3i_2_3")

					remove_vehicles_3_S = traci.inductionloop.getLastStepVehicleIDs("e1Detector_1o_1_5")
					remove_vehicles_3_L = traci.inductionloop.getLastStepVehicleIDs("e1Detector_2o_2_6")


					add_vehicles_4_S = traci.inductionloop.getLastStepVehicleIDs("e1Detector_4i_1_4")
					add_vehicles_4_L = traci.inductionloop.getLastStepVehicleIDs("e1Detector_4i_2_4")

					remove_vehicles_4_S = traci.inductionloop.getLastStepVehicleIDs("e1Detector_2o_1_6")
					remove_vehicles_4_L = traci.inductionloop.getLastStepVehicleIDs("e1Detector_3o_2_7")


					add_vehicles(add_vehicles_1_L)
					add_vehicles(add_vehicles_1_S)

					add_vehicles(add_vehicles_2_L)
					add_vehicles(add_vehicles_2_S)

					add_vehicles(add_vehicles_3_L)
					add_vehicles(add_vehicles_3_S)

					add_vehicles(add_vehicles_4_L)
					add_vehicles(add_vehicles_4_S)

					remove_vehicles(remove_vehicles_1_L)
					remove_vehicles(remove_vehicles_1_S)

					remove_vehicles(remove_vehicles_2_L)
					remove_vehicles(remove_vehicles_2_S)

					remove_vehicles(remove_vehicles_3_L)
					remove_vehicles(remove_vehicles_3_S)

					remove_vehicles(remove_vehicles_4_L)
					remove_vehicles(remove_vehicles_4_S)

					for vehicle1_S in add_vehicles_1_S:
						if vehicle1_S not in vehicles_in_loop_1_S:
							vehicles_in_loop_1_S.append(vehicle1_S)
						if vehicle1_S not in vehicles_waiting_1_S:
							vehicles_waiting_1_S.append(vehicle1_S)
							if position1_S:
								i1 = len(position1_S)
								tmp_pos1_S = position1_S[i1 - 1]
								position1_S.append(tmp_pos1_S - 3)
								traci.vehicle.setStop(vehicle1_S, "1i", position1_S[i1], 1)
							else:
								for i1 in range(0, len(vehicles_waiting_1_S)):
									position1_S.append(485 - 3 * i1)
									traci.vehicle.setStop(vehicle1_S, "1i", position1_S[i1], 1)


					for vehicle1_L in add_vehicles_1_L:
						if vehicle1_L not in vehicles_in_loop_1_L:
							vehicles_in_loop_1_L.append(vehicle1_L)
						if vehicle1_L not in vehicles_waiting_1_L:
							vehicles_waiting_1_L.append(vehicle1_L)
							if position1_L:
								i3 = len(position1_L)
								tmp_pos1_L = position1_L[i3 - 1]
								position1_L.append(tmp_pos1_L - 3)
								traci.vehicle.setStop(vehicle1_L, "1i", position1_L[i3], 2)
							else:
								for i3 in range(0, len(vehicles_waiting_1_L)):
									position1_L.append(485 - 3 * i3)
									traci.vehicle.setStop(vehicle1_L, "1i", position1_L[i3], 2)


					for vehicle2_S in add_vehicles_2_S:
						if vehicle2_S not in vehicles_in_loop_2_S:
							vehicles_in_loop_2_S.append(vehicle2_S)
						if vehicle2_S not in vehicles_waiting_2_S:
							vehicles_waiting_2_S.append(vehicle2_S)
							if position2_S:
								j1 = len(position2_S)
								tmp_pos2_S = position2_S[j1 - 1]
								position2_S.append(tmp_pos2_S - 3)
								traci.vehicle.setStop(vehicle2_S, "2i", position2_S[j1], 1)
							else:
								for j1 in range(0, len(vehicles_waiting_2_S)):
									position2_S.append(485 - 3 * j1)
									traci.vehicle.setStop(vehicle2_S, "2i", position2_S[j1], 1)


					for vehicle2_L in add_vehicles_2_L:
						if vehicle2_L not in vehicles_in_loop_2_L:
							vehicles_in_loop_2_L.append(vehicle2_L)
						if vehicle2_L not in vehicles_waiting_2_L:
							vehicles_waiting_2_L.append(vehicle2_L)
							if position2_L:
								j3 = len(position2_L)
								tmp_pos2_L = position2_L[j3 - 1]
								position2_L.append(tmp_pos2_L - 3)
								traci.vehicle.setStop(vehicle2_L, "2i", position2_L[j3], 2)
							else:
								for j3 in range(0, len(vehicles_waiting_2_L)):
									position2_L.append(485 - 3 * j3)
									traci.vehicle.setStop(vehicle2_L, "2i", position2_L[j3], 2)


					for vehicle3_S in add_vehicles_3_S:
						if vehicle3_S not in vehicles_in_loop_3_S:
							vehicles_in_loop_3_S.append(vehicle3_S)
						if vehicle3_S not in vehicles_waiting_3_S:
							vehicles_waiting_3_S.append(vehicle3_S)
							if position3_S:
								k1 = len(position3_S)
								tmp_pos3_S = position3_S[k1 - 1]
								position3_S.append(tmp_pos3_S - 3)
								traci.vehicle.setStop(vehicle3_S, "3i", position3_S[k1], 1)
							else:
								for k1 in range(0, len(vehicles_waiting_3_S)):
									position3_S.append(485 - 3 * k1)
									traci.vehicle.setStop(vehicle3_S, "3i", position3_S[k1], 1)


					for vehicle3_L in add_vehicles_3_L:
						if vehicle3_L not in vehicles_in_loop_3_L:
							vehicles_in_loop_3_L.append(vehicle3_L)
						if vehicle3_L not in vehicles_waiting_3_L:
							vehicles_waiting_3_L.append(vehicle3_L)
							if position3_L:
								k3 = len(position3_L)
								tmp_pos3_L = position3_L[k3 - 1]
								position3_L.append(tmp_pos3_L - 3)
								traci.vehicle.setStop(vehicle3_L, "3i", position3_L[k3], 2)
							else:
								for k3 in range(0, len(vehicles_waiting_3_L)):
									position3_L.append(485 - 3 * k3)
									traci.vehicle.setStop(vehicle3_L, "3i", position3_L[k3], 2)


					for vehicle4_S in add_vehicles_4_S:
						if vehicle4_S not in vehicles_in_loop_4_S:
							vehicles_in_loop_4_S.append(vehicle4_S)
						if vehicle4_S not in vehicles_waiting_4_S:
							vehicles_waiting_4_S.append(vehicle4_S)
							if position4_S:
								l1 = len(position4_S)
								tmp_pos4_S = position4_S[l1 - 1]
								position4_S.append(tmp_pos4_S - 3)
								traci.vehicle.setStop(vehicle4_S, "4i", position4_S[l1], 1)
							else:
								for l1 in range(0, len(vehicles_waiting_4_S)):
									position4_S.append(485 - 3 * l1)
									traci.vehicle.setStop(vehicle4_S, "4i", position4_S[l1], 1)


					for vehicle4_L in add_vehicles_4_L:
						if vehicle4_L not in vehicles_in_loop_4_L:
							vehicles_in_loop_4_L.append(vehicle4_L)
						if vehicle4_L not in vehicles_waiting_4_L:
							vehicles_waiting_4_L.append(vehicle4_L)
							if position4_L:
								l3 = len(position4_L)
								tmp_pos4_L = position4_L[l3 - 1]
								position4_L.append(tmp_pos4_L - 3)
								traci.vehicle.setStop(vehicle4_L, "4i", position4_L[l3], 2)
							else:
								for l3 in range(0, len(vehicles_waiting_4_L)):
									position4_L.append(485 - 3 * l3)
									traci.vehicle.setStop(vehicle4_L, "4i", position4_L[l3], 2)

					if remove_vehicles_1_S:
						if remove_vehicles_1_S[0] in vehicles_in_loop_1_S:
							vehicles_in_loop_1_S.pop(0)

					if remove_vehicles_1_L:
						if remove_vehicles_1_L[0] in vehicles_in_loop_1_L:
							vehicles_in_loop_1_L.pop(0)

					if remove_vehicles_2_S:
						if remove_vehicles_2_S[0] in vehicles_in_loop_2_S:
							vehicles_in_loop_2_S.pop(0)

					if remove_vehicles_2_L:
						if remove_vehicles_2_L[0] in vehicles_in_loop_2_L:
							vehicles_in_loop_2_L.pop(0)

					if remove_vehicles_3_S:
						if remove_vehicles_3_S[0] in vehicles_in_loop_3_S:
							vehicles_in_loop_3_S.pop(0)

					if remove_vehicles_3_L:
						if remove_vehicles_3_L[0] in vehicles_in_loop_3_L:
							vehicles_in_loop_3_L.pop(0)

					if remove_vehicles_4_S:
						if remove_vehicles_4_S[0] in vehicles_in_loop_4_S:
							vehicles_in_loop_4_S.pop(0)

					if remove_vehicles_4_L:
						if remove_vehicles_4_L[0] in vehicles_in_loop_4_L:
							vehicles_in_loop_4_L.pop(0)


					S_ = set_state_space(S_, vehicles_in_loop_1_L, vehicles_in_loop_1_S, vehicles_in_loop_2_L,
									vehicles_in_loop_2_S, vehicles_in_loop_3_L, vehicles_in_loop_3_S,
									vehicles_in_loop_4_L, vehicles_in_loop_4_S)
									
					R = len(remove_vehicles_1S) + len(remove_vehicles_2S) + len(remove_vehicles_3S) + len(remove_vehicles_4S) + len(remove_vehicles_1L) 
					+ len(remove_vehicles_2L) + len(remove_vehicles_3L) + len(remove_vehicles_4L)

					R_ = traci.simulation.getCollidingVehiclesNumber()
					R_List = traci.simulation.getCollidingVehiclesIDList()
					
					for vehicleC in R_List:
						if vehicleC not in collided_vehicles:
							collided_vehicles.append(vehicleC)
							list_is_new = True

					if R_ and list_is_new == True:

						R_collision = 1
						collision_reward = -1
						R_ = R_ * collision_reward / 2.5

					total_reward += R + R_
					step += 1

					for i in range (0, len(vehicles_in_loop_1_L)):
						co2_1L = 0
						co2_1L = traci.vehicle.getCO2Emission(vehicles_in_loop_1_L[i])
						Tot_co2_1L += co2_1L

					if vehicles_waiting_1_L:
						for j in range(0, len(vehicles_waiting_1_L)):
							wt_1L = traci.vehicle.getWaitingTime(vehicles_waiting_1_L[j])
							Tot_wt_time += wt_1L

					for i in range (0, len(vehicles_in_loop_1_S)):
						co2_1S = 0
						co2_1S = traci.vehicle.getCO2Emission(vehicles_in_loop_1_S[i])
						Tot_co2_1S += co2_1S

					if vehicles_waiting_1_S:
						for j in range(0, len(vehicles_waiting_1_S)):
							wt_1S = traci.vehicle.getWaitingTime(vehicles_waiting_1_S[j])
							Tot_wt_time += wt_1S

					for i in range(0, len(vehicles_in_loop_2_L)):
						co2_2L = 0
						co2_2L = traci.vehicle.getCO2Emission(vehicles_in_loop_2_L[i])
						Tot_co2_2L += co2_2L

					if vehicles_waiting_2_L:
						for j in range(0, len(vehicles_waiting_2_L)):
							wt_2L = traci.vehicle.getWaitingTime(vehicles_waiting_2_L[j])
							Tot_wt_time += wt_2L

					for i in range(0, len(vehicles_in_loop_2_S)):
						co2_2S = 0
						co2_2S = traci.vehicle.getCO2Emission(vehicles_in_loop_2_S[i])
						Tot_co2_2S += co2_2S

					if vehicles_waiting_2_S:
						for j in range(0, len(vehicles_waiting_2_S)):
							wt_2S = traci.vehicle.getWaitingTime(vehicles_waiting_2_S[j])
							Tot_wt_time += wt_2S

					for i in range(0, len(vehicles_in_loop_3_L)):
						co2_3L = 0
						co2_3L = traci.vehicle.getCO2Emission(vehicles_in_loop_3_L[i])
						Tot_co2_3L += co2_3L

					if vehicles_waiting_3_L:
						for j in range(0, len(vehicles_waiting_3_L)):
							wt_3L = traci.vehicle.getWaitingTime(vehicles_waiting_3_L[j])
							Tot_wt_time += wt_3L

					for i in range(0, len(vehicles_in_loop_3_S)):
						co2_3S = 0
						co2_3S = traci.vehicle.getCO2Emission(vehicles_in_loop_3_S[i])
						Tot_co2_3S += co2_3S

					if vehicles_waiting_3_S:
						for j in range(0, len(vehicles_waiting_3_S)):
							wt_3S = traci.vehicle.getWaitingTime(vehicles_waiting_3_S[j])
							Tot_wt_time += wt_3S

					for i in range(0, len(vehicles_in_loop_4_S)):
						co2_4S = 0
						co2_4S = traci.vehicle.getCO2Emission(vehicles_in_loop_4_S[i])
						Tot_co2_4S += co2_4S

					if vehicles_waiting_4_S:
						for j in range(0, len(vehicles_waiting_4_S)):
							wt_4S = traci.vehicle.getWaitingTime(vehicles_waiting_4_S[j])
							Tot_wt_time += wt_4S

					for i in range(0, len(vehicles_in_loop_4_L)):
						co2_4L = 0
						co2_4L = traci.vehicle.getCO2Emission(vehicles_in_loop_4_L[i])
						Tot_co2_4L += co2_4L

					if vehicles_waiting_4_L:
						for j in range(0, len(vehicles_waiting_4_L)):
							wt_4L = traci.vehicle.getWaitingTime(vehicles_waiting_4_L[j])
							Tot_wt_time += wt_4L


					total_co2_emission += Tot_co2_1L + Tot_co2_1S + Tot_co2_2L + Tot_co2_2S + Tot_co2_3L + Tot_co2_3S + Tot_co2_4S + Tot_co2_4L
					waiting_time += wt_time_1L + wt_time_1S + wt_time_2L + wt_time_2S + wt_time_3L + wt_time_3S + wt_time_4L + wt_time_4S

					S = S_

            traci.close()
            sys.stdout.flush()

            managed_vehicles = []

        print("Finish")


def set_state_space(S, a, b, c, d, e, f, g, h):

    if a:
        plc1 = traci.vehicle.getLanePosition(a[0])
        np.put(S, 0, round(plc1) / 10)
    else:
        np.put(S, 0, 0)

    if b:
        plc2 = traci.vehicle.getLanePosition(b[0])
        np.put(S, 1, round(plc2) / 10)
    else:
        np.put(S, 1, 0)

    if c:
        plc3 = traci.vehicle.getLanePosition(c[0])
        np.put(S, 2, round(plc3) / 10)
    else:
        np.put(S, 2, 0)

    if d:
        plc4 = traci.vehicle.getLanePosition(d[0])
        np.put(S, 3, round(plc4) / 10)
    else:
        np.put(S, 3, 0)

    if e:
        plc5 = traci.vehicle.getLanePosition(e[0])
        np.put(S, 4, round(plc5) / 10)
    else:
        np.put(S, 4, 0)

    if f:
        plc6 = traci.vehicle.getLanePosition(f[0])
        np.put(S, 5, round(plc6) / 10)
    else:
        np.put(S, 5, 0)

    if g:
        plc7 = traci.vehicle.getLanePosition(g[0])
        np.put(S, 6, round(plc7) / 10)
    else:
        np.put(S, 6, 0)

    if h:
        plc8 = traci.vehicle.getLanePosition(h[0])
        np.put(S, 7, round(plc8) / 10)
    else:
        np.put(S, 7, 0)

    return S

def add_vehicles(vehicles):
    global managed_vehicles
    for vehicle in vehicles:
        if not vehicle in managed_vehicles:
            managed_vehicles.append(vehicle)

def remove_vehicles(vehicles):
    global managed_vehicles
    for vehicle in vehicles:
        if vehicle in managed_vehicles:
            managed_vehicles.remove(vehicle)


def generate_routefile():
    random.seed(42)  # make tests reproducible
    N = 50000  # number of time steps
    # demand per second from different directions
    traffic_density = [random.randint(-3, 3) for i in range (0, 8)]
    pWE = 1. / (15 - traffic_density[0]) #12
    pWN = 1. / (15 - traffic_density[1]) #22
    pWS = 1. / (15) #24
    pEW = 1. / (15 - traffic_density[2]) #13
    pEN = 1. / (15) #26
    pES = 1. / (15 - traffic_density[3]) #28
    pNS = 1. / (15 - traffic_density[4]) #30
    pNE = 1. / (15) #32
    pNW = 1. / (15 - traffic_density[5]) #34
    pSN = 1. / (15 - traffic_density[6]) #36
    pSE = 1. / (15) #38
    pSW = 1. / (15 - traffic_density[7]) #40

    with open("data/cross.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="typeWE" accel="0.6" decel="6.5" sigma="0.5" length="5" minGap="1.5" maxSpeed="10" guiShape="passenger" />
        <vType id="typeNS" accel="0.6" decel="6.5" sigma="0.5" length="5" minGap="1.5" maxSpeed="10" guiShape="passenger" />
        <route id="right" edges="51o 1i 3o 52i" />
        <route id="right_up" edges="51o 1i 4o 54i" />
        <route id="right_down" edges="51o 1i 2o 53i" />
        <route id="left" edges="52o 3i 1o 51i" />
        <route id="left_up" edges="52o 3i 4o 54i" />
        <route id="left_down" edges="52o 3i 2o 53i" />
        <route id ="up" edges="53o 2i 4o 54i" />
        <route id ="up_left" edges="53o 2i 1o 51i" />
        <route id ="up_right" edges="53o 2i 3o 52i" />
        <route id ="down_right" edges="54o 4i 1o 51i" />
        <route id ="down_left" edges="54o 4i 3o 52i" />
        <route id="down" edges="54o 4i 2o 53i" />""", file=routes)
        vehNr = 0
        for i in range(N):
            if random.uniform(0, 1) < pWE:  # % (vehNr, i) is to pass it as a tuple
                print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                continue
            if random.uniform(0, 1) < pWN:
                print('    <vehicle id="rightU_%i" type="typeWE" route="right_up" depart="%i" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                continue
            if random.uniform(0, 1) < pWS:
                print('    <vehicle id="rightD_%i" type="typeWE" route="right_down" depart="%i" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                continue
            if random.uniform(0, 1) < pEW:
                print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                continue
            if random.uniform(0, 1) < pEN:
                print('    <vehicle id="leftU_%i" type="typeWE" route="left_up" depart="%i" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                continue
            if random.uniform(0, 1) < pES:
                print('    <vehicle id="leftD_%i" type="typeWE" route="left_down" depart="%i" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                continue
            if random.uniform(0, 1) < pSN:
                print('    <vehicle id="up_%i" type="typeNS" route="up" depart="%i" color="1,0,0" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                continue
            if random.uniform(0, 1) < pSE:
                print('    <vehicle id="upR_%i" type="typeNS" route="up_right" depart="%i" color="1,0,0" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                continue
            if random.uniform(0, 1) < pSW:
                print('    <vehicle id="upL_%i" type="typeNS" route="up_left" depart="%i" color="1,0,0" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                continue
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="down_%i" type="typeNS" route="down" depart="%i" color="1,0,0" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                continue
            if random.uniform(0, 1) < pNE:
                print('    <vehicle id="downR_%i" type="typeNS" route="down_right" depart="%i" color="1,0,0" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                continue
            if random.uniform(0, 1) < pNW:
                print('    <vehicle id="downL_%i" type="typeNS" route="down_left" depart="%i" color="1,0,0" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                continue
        print("</routes>", file=routes)

def get_options():
    optParser = optparse. OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

if __name__ == '__main__':

    q_table = playGame()
    print("Finished the Process")




