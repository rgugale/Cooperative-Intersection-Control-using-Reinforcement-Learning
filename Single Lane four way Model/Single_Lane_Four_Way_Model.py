from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random
import numpy as np
import collections as col
import tensorflow as tf
import re
from itertools import chain
import ReplayBuffer

try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")


import traci

tf.reset_default_graph()

N_ACTIONS = 4
ACTIONS = ['Lane1', 'Lane2', 'Lane3', 'Lane4']


EPSILON_Orig = 0.7 # Greedy Policy
ALPHA = 0.001 # Learning Rate
GAMMA = 0.9
MAX_EPISODES = 25
MAX_STEPS = 3000
MAX_EXPLORATION_STEPS = 10000

PRE_TRAIN_STEPS = 15000
update_frequency = 2000
batch_size = 64

step = 0
LIST_STATES = []

np.random.seed(1337)


input_NN = tf.placeholder(shape=[1, 8], dtype=tf.float64)
weights1 = tf.Variable(tf.random_uniform([8, 8], 0, 0.01, dtype=tf.float64))
bias1 = tf.Variable(tf.zeros([8], dtype=tf.float64))
output1 = tf.nn.relu(tf.matmul(input_NN, weights1) + bias1)

weights2 = tf.Variable(tf.random_uniform([8, 4], 0, 0.01, dtype=tf.float64))
Qout = tf.nn.softmax(tf.matmul(output1, weights2))
predict = tf.argmax(Qout, 1)

nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float64)
loss = tf.square(nextQ - Qout)
trainer = tf.train.GradientDescentOptimizer(learning_rate=ALPHA)
updateModel= trainer.minimize(loss)

global managed_vehicles
managed_vehicles = []

def playGame():

    global managed_vehicles

    observations = col.namedtuple('observation', 'managed_vehicles')

    # SUMO Stuff
    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    init = tf.initialize_all_variables()
    EPSILON = EPSILON_Orig

    myBuffer = ReplayBuffer.experience_buffer()


    with tf.Session() as sess:
        sess.run(init)

        global_step = 0

        for episode in range(MAX_EPISODES):

            if episode < MAX_EPISODES - 1:
                train = True
                test = False
            else:
                train = False
                test = True

            episodeBuffer = ReplayBuffer.experience_buffer()
			
            vehicles_waiting_1 = []
            vehicles_waiting_2 = []
            vehicles_waiting_3 = []
            vehicles_waiting_4 = []
            vehicles_in_loop_1 = []
            vehicles_in_loop_2 = []
            vehicles_in_loop_3 = []
            vehicles_in_loop_4 = []

            position1 = []
            position2 = []
            position3 = []
            position4 = []

            disabled_speed_vehicle_ids = []
            running_vehicle_ids = []
            collided_vehicles = []

            generate_routefile()

            waiting_time = 0

            traci.start([sumoBinary, "-c", "data/cross.sumocfg",
                         "--tripinfo-output", "tripinfo.xml", "--additional-files",
                         "data/cross.additionals.xml",
                         "--collision.check-junctions",
                         "--collision.mingap-factor", "1",
                         "--collision.action", "warn",
                         "--step-length", "1",
                         "--error-log", "error.txt"])

            observations.managed_vehicles = 0

            S = np.zeros([1, 8], dtype=float)
            A = 'Lane1'
            B = np.zeros(1)

            total_reward = 0

            for step in range(MAX_STEPS):

                if train:

                    R_ = 0
                    R = 0
                    R_collision = 0
                    wt_time_1 = 0
                    wt_time_2 = 0
                    wt_time_3 = 0
                    wt_time_4 = 0            
                    list_is_new = False

                    running_vehicle_ids = traci.vehicle.getIDList()
                    for vehicleID in running_vehicle_ids:
                        if vehicleID not in disabled_speed_vehicle_ids:

                            traci.vehicle.setLaneChangeMode(vehicleID, 0b0100000000)
                            disabled_speed_vehicle_ids.append(vehicleID)

                    S_old = S.copy()
                    S_col = S_old.copy()
                    A_old = A
                    B_old = B.copy()

                    B, allQ = sess.run([predict, Qout], feed_dict={input_NN:S})

                    S_ = S.copy()

                    if B == 0:
                        A ='Lane1'

                    if B == 1:
                        A ='Lane2'

                    if B == 2:
                        A ='Lane3'

                    if B == 3:
                        A ='Lane4'

                    if (np.random.uniform() > EPSILON):
                        A = np.random.choice(ACTIONS)

                    if (A == 'Lane1' or (np.equal(S[0, 0], 1) and np.equal(S[0, 2], 0) and np.equal(S[0, 4], 0) and np.equal(S[0, 6], 0))
                            or (np.equal(S[0, 0], 1) and np.equal(S[0, 2], 0) and np.equal(S[0, 4], 1) and np.equal(S[0, 6], 0))):
                        if vehicles_waiting_1:
                            # ADD GET POSITION OF 1st VEHICLE
                            traci.vehicle.setStop(vehicles_waiting_1[0], "1i", position1[0], 0, 0)
                            traci.vehicle.setSpeed(vehicles_waiting_1[0], 10)
                            traci.vehicle.setSpeedMode(vehicles_waiting_1[0], 0)
                            wt_time_1 = traci.vehicle.getWaitingTime(vehicles_waiting_1[0])
                            vehicles_waiting_1.pop(0)
                            position1.pop(0)
                            A = 'Lane1'
                            B[0] = 0

                    if (A == 'Lane2' or (np.equal(S[0, 0], 0) and np.equal(S[0, 2], 1) and np.equal(S[0, 4], 0) and np.equal(S[0, 6], 0))
                            or (np.equal(S[0, 0], 0) and np.equal(S[0, 2], 1) and np.equal(S[0, 4], 0) and np.equal(S[0, 6], 1))):
                        if vehicles_waiting_2:
                            # ADD GET POSITION OF 1st VEHICLE
                            traci.vehicle.setStop(vehicles_waiting_2[0], "3i", position2[0], 0, 0)
                            traci.vehicle.setSpeed(vehicles_waiting_2[0], 10)
                            traci.vehicle.setSpeedMode(vehicles_waiting_2[0], 0)
                            wt_time_2 = traci.vehicle.getWaitingTime(vehicles_waiting_2[0])
                            vehicles_waiting_2.pop(0)
                            position2.pop(0)
                            A = 'Lane2'
                            B[0] = 1

                    if (A == 'Lane3' or (np.equal(S[0, 0], 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 4], 1) and np.equal(S[0, 6], 0))
                            or (np.equal(S[0, 0], 1) and np.equal(S[0, 2], 0) and np.equal(S[0, 4], 1) and np.equal(S[0, 6], 0))):
                        if vehicles_waiting_3:
                            # ADD GET POSITION OF 1st VEHICLE
                            traci.vehicle.setStop(vehicles_waiting_3[0], "2i", position3[0], 0, 0)
                            traci.vehicle.setSpeed(vehicles_waiting_3[0], 10)
                            traci.vehicle.setSpeedMode(vehicles_waiting_3[0], 0)
                            wt_time_3 = traci.vehicle.getWaitingTime(vehicles_waiting_3[0])
                            vehicles_waiting_3.pop(0)
                            position3.pop(0)
                            A = 'Lane3'
                            B[0] = 2

                    if (A == 'Lane4' or (np.equal(S[0, 0], 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 4], 0) and np.equal(S[0, 6], 1))
                            or (np.equal(S[0, 0], 0) and np.equal(S[0, 2], 1) and np.equal(S[0, 4], 0) and np.equal(S[0, 6], 1))):
                        if vehicles_waiting_4:
                            # ADD GET POSITION OF 1st VEHICLE
                            traci.vehicle.setStop(vehicles_waiting_4[0], "4i", position4[0], 0, 0)
                            traci.vehicle.setSpeed(vehicles_waiting_4[0], 10)
                            traci.vehicle.setSpeedMode(vehicles_waiting_4[0], 0)
                            wt_time_4 = traci.vehicle.getWaitingTime(vehicles_waiting_4[0])
                            vehicles_waiting_4.pop(0)
                            position4.pop(0)
                            A = 'Lane4'
                            B[0] = 3

                    if 495 not in position1:
                        if vehicles_waiting_1:
                            pos1 = 495
                            i1 = 0
                            except1 = 0
                            change_position1 = []
                            for vehicle1 in vehicles_waiting_1:
                                try:
                                    traci.vehicle.setStop(vehicle1, "1i", position1[i1], 0, 0)
                                    traci.vehicle.setStop(vehicle1, "1i", pos1)
                                    change_position1.append(pos1)
                                    pos1 -= 3
                                    i1 += 1
                                except Exception:
                                    except1 = 1
                                    continue
                            if except1:
                                traci.vehicle.remove(vehicles_waiting_1[0], reason=3)
                                if vehicles_waiting_1[0] in managed_vehicles:
                                    managed_vehicles.remove(vehicles_waiting_1[0])
                                    vehicles_in_loop_1.remove(vehicles_waiting_1[0])
                                if vehicles_waiting_1[0] in running_vehicle_ids:
                                    running_vehicle_ids.remove(vehicles_waiting_1[0])
                                vehicles_waiting_1.pop(0)
                                position1.pop(0)
                            else:
                                position1 = change_position1

                    if 495 not in position2:
                        if vehicles_waiting_2:
                            pos2 = 495
                            i2 = 0
                            except2 = 0
                            change_position2 = []
                            for vehicle2 in vehicles_waiting_2:
                                try:
                                    traci.vehicle.setStop(vehicle2, "3i", position2[i2], 0, 0)
                                    traci.vehicle.setStop(vehicle2, "3i", pos2)
                                    change_position2.append(pos2)
                                    pos2 -= 3
                                    i2 += 1
                                except Exception:
                                    except2 = 1
                                    continue
                            if except2:
                                traci.vehicle.remove(vehicles_waiting_2[0], reason=3)
                                if vehicles_waiting_2[0] in managed_vehicles:
                                    managed_vehicles.remove(vehicles_waiting_2[0])
                                    vehicles_in_loop_2.remove(vehicles_waiting_2[0])
                                if vehicles_waiting_2[0] in running_vehicle_ids:
                                    running_vehicle_ids.remove(vehicles_waiting_2[0])
                                vehicles_waiting_2.pop(0)
                                position2.pop(0)
                            else:
                                position2 = change_position2

                    if 495 not in position3:
                        if vehicles_waiting_3:
                            pos3 = 495
                            i3 = 0
                            except3 = 0
                            change_position3 = []
                            for vehicle3 in vehicles_waiting_3:
                                try:
                                    traci.vehicle.setStop(vehicle3, "2i", position3[i3], 0, 0)
                                    traci.vehicle.setStop(vehicle3, "2i", pos3)
                                    change_position3.append(pos3)
                                    pos3 -= 3
                                    i3 += 1
                                except Exception:
                                    except3 = 1
                                    continue
                            if except3:
                                traci.vehicle.remove(vehicles_waiting_3[0], reason=3)
                                if vehicles_waiting_3[0] in managed_vehicles:
                                    managed_vehicles.remove(vehicles_waiting_3[0])
                                    vehicles_in_loop_3.remove(vehicles_waiting_3[0])
                                if vehicles_waiting_3[0] in running_vehicle_ids:
                                    running_vehicle_ids.remove(vehicles_waiting_3[0])
                                vehicles_waiting_3.pop(0)
                                position3.pop(0)
                            else:
                                position3 = change_position3

                    if 495 not in position4:
                        if vehicles_waiting_4:
                            pos4 = 495
                            i4 = 0
                            except4 = 0
                            change_position4 = []
                            for vehicle4 in vehicles_waiting_4:
                                try:
                                    traci.vehicle.setStop(vehicle4, "4i", position4[i4], 0, 0)
                                    traci.vehicle.setStop(vehicle4, "4i", pos4)
                                    change_position4.append(pos4)
                                    pos4 -= 3
                                    i4 += 1
                                except Exception:
                                    except4 = 1
                                    continue
                            if except4:
                                traci.vehicle.remove(vehicles_waiting_4[0], reason=3)
                                if vehicles_waiting_4[0] in managed_vehicles:
                                    managed_vehicles.remove(vehicles_waiting_4[0])
                                    vehicles_in_loop_4.remove(vehicles_waiting_4[0])
                                if vehicles_waiting_4[0] in running_vehicle_ids:
                                    running_vehicle_ids.remove(vehicles_waiting_4[0])
                                vehicles_waiting_4.pop(0)
                                position4.pop(0)
                            else:
                                position4 = change_position4

                    traci.simulationStep()

                    add_vehicles_1 = traci.inductionloop.getLastStepVehicleIDs("e1Detector_1i_0_0")
                    remove_vehicles_1 = traci.inductionloop.getLastStepVehicleIDs("e1Detector_2o_0_5")
                    add_vehicles_3 = traci.inductionloop.getLastStepVehicleIDs("e1Detector_2i_0_2")
                    remove_vehicles_3 = traci.inductionloop.getLastStepVehicleIDs("e1Detector_1o_0_6")
                    add_vehicles_2 = traci.inductionloop.getLastStepVehicleIDs("e1Detector_3i_0_1")
                    remove_vehicles_2 = traci.inductionloop.getLastStepVehicleIDs("e1Detector_4o_0_7")
                    add_vehicles_4 = traci.inductionloop.getLastStepVehicleIDs("e1Detector_4i_0_3")
                    remove_vehicles_4 = traci.inductionloop.getLastStepVehicleIDs("e1Detector_3o_0_4")

                    add_vehicles(add_vehicles_1)
                    add_vehicles(add_vehicles_2)
                    add_vehicles(add_vehicles_3)
                    add_vehicles(add_vehicles_4)

                    remove_vehicles(remove_vehicles_1)
                    remove_vehicles(remove_vehicles_2)
                    remove_vehicles(remove_vehicles_3)
                    remove_vehicles(remove_vehicles_4)

                    for vehicle1 in add_vehicles_1:
                        if vehicle1 not in vehicles_in_loop_1:
                            vehicles_in_loop_1.append(vehicle1)
                        if vehicle1 not in vehicles_waiting_1:
                            vehicles_waiting_1.append(vehicle1)
                            if position1:
                                i = len(position1)
                                tmp_pos1 = position1[i - 1]
                                position1.append(tmp_pos1 - 3)
                                traci.vehicle.setStop(vehicle1, "1i", position1[i])
                            else:
                                for i in range(0, len(vehicles_waiting_1)):
                                    position1.append(495 - 3 * i)
                                    traci.vehicle.setStop(vehicle1, "1i", position1[i])

                    for vehicle3 in add_vehicles_3:
                        if vehicle3 not in vehicles_in_loop_3:
                            vehicles_in_loop_3.append(vehicle3)
                        if vehicle3 not in vehicles_waiting_3:
                            vehicles_waiting_3.append(vehicle3)
                            if position3:
                                j = len(position3)
                                tmp_pos3 = position3[j - 1]
                                position3.append(tmp_pos3 - 3)
                                traci.vehicle.setStop(vehicle3, "2i", position3[j])
                            else:
                                for j in range(0, len(vehicles_waiting_3)):
                                    position3.append(495 - 3 * j)
                                    traci.vehicle.setStop(vehicle3, "2i", position3[j])

                    for vehicle2 in add_vehicles_2:
                        if vehicle2 not in vehicles_in_loop_2:
                            vehicles_in_loop_2.append(vehicle2)
                        if vehicle2 not in vehicles_waiting_2:
                            vehicles_waiting_2.append(vehicle2)
                            if position2:
                                k = len(position2)
                                tmp_pos2 = position2[k - 1]
                                position2.append(tmp_pos2 - 3)
                                traci.vehicle.setStop(vehicle2, "3i", position2[k])
                            else:
                                for k in range(0, len(vehicles_waiting_2)):
                                    position2.append(495 - 3 * k)
                                    traci.vehicle.setStop(vehicle2, "3i", position2[k])

                    for vehicle4 in add_vehicles_4:
                        if vehicle4 not in vehicles_in_loop_4:
                            vehicles_in_loop_4.append(vehicle4)
                        if vehicle4 not in vehicles_waiting_4:
                            vehicles_waiting_4.append(vehicle4)
                            if position4:
                                l = len(position4)
                                tmp_pos4 = position4[l - 1]
                                position4.append(tmp_pos4 - 3)
                                traci.vehicle.setStop(vehicle4, "4i", position4[l])
                            else:
                                for l in range(0, len(vehicles_waiting_4)):
                                    position4.append(495 - 3 * l)
                                    traci.vehicle.setStop(vehicle4, "4i", position4[l])

                    if remove_vehicles_1:
                        if remove_vehicles_1[0] in vehicles_in_loop_1:
                            vehicles_in_loop_1.pop(0)

                    if remove_vehicles_2:
                        if remove_vehicles_2[0] in vehicles_in_loop_2:
                            vehicles_in_loop_2.pop(0)

                    if remove_vehicles_3:
                        if remove_vehicles_3[0] in vehicles_in_loop_3:
                            vehicles_in_loop_3.pop(0)

                    if remove_vehicles_4:
                        if remove_vehicles_4[0] in vehicles_in_loop_4:
                            vehicles_in_loop_4.pop(0)


                    S_ = set_state_space(S_, vehicles_in_loop_1, vehicles_in_loop_2, vehicles_in_loop_3, vehicles_in_loop_4)

                    R = len(remove_vehicles_1) + len(remove_vehicles_2) + len(remove_vehicles_3) + len(remove_vehicles_4)

                    if remove_vehicles_1:
                        A = 'Lane1'
                        B[0] = 0

                    if remove_vehicles_2:
                        A = 'Lane2'
                        B[0] = 1

                    if remove_vehicles_3:
                        A = 'Lane3'
                        B[0] = 2

                    if remove_vehicles_4:
                        A = 'Lane4'
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
                        R_ = R_ * collision_reward / 2.75

                        c = len(collided_vehicles)
                        d1 = collided_vehicles[c - 1]
                        d2 = collided_vehicles[c - 2]

                        m = ['right', 'up', 'left', 'down']

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
                                    pos1 = S_old[0, 1]
                                    n1 = 0
                                    break

                                if n == 1:
                                    pos1 = S_old[0, 3]
                                    n1 = 1
                                    break

                                if n == 2:
                                    pos1 = S_old[0, 5]
                                    n1 = 2
                                    break

                                if n == 3:
                                    pos1 = S_old[0, 7]
                                    n1 = 3
                                    break

                        for n in range(len(m)):
                            m[n] = re.compile(m[n])
                            p = m[n].match(d2)
                            if p:
                                if n == 0:
                                    pos2 = S_old[0, 1]
                                    n2 = 0
                                    break

                                if n == 1:
                                    pos2 = S_old[0, 3]
                                    n2 = 1
                                    break

                                if n == 2:
                                    pos2 = S_old[0, 5]
                                    n2 = 2
                                    break

                                if n == 3:
                                    pos2 = S_old[0, 7]
                                    n2 = 3
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
                            A_old = 'Lane1'
                            B_old[0] = 0

                        if col_action == 1:
                            A_old = 'Lane2'
                            B_old[0] = 1

                        if col_action == 2:
                            A_old = 'Lane3'
                            B_old[0] = 2

                        if col_action == 3:
                            A_old = 'Lane4'
                            B_old[0] = 3


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

                    EPSILON = min(EPSILON + (1 - EPSILON_Orig) / (MAX_EXPLORATION_STEPS), 0.99)

                    if step > PRE_TRAIN_STEPS:
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

                    waiting_time += wt_time_1 + wt_time_2 + wt_time_3 + wt_time_4

                    global_step += 1

                if test:

                    R_ = 0
                    R = 0
                    R_collision = 0
                    wt_time_1 = 0
                    wt_time_2 = 0
                    wt_time_3 = 0
                    wt_time_4 = 0
                    list_is_new = False

                    running_vehicle_ids = traci.vehicle.getIDList()
                    for vehicleID in running_vehicle_ids:
                        if vehicleID not in disabled_speed_vehicle_ids:
                            traci.vehicle.setLaneChangeMode(vehicleID, 0b0100000000)
                            disabled_speed_vehicle_ids.append(vehicleID)

                    S_old = S.copy()
                    S_col = S_old.copy()
                    A_old = A
                    B_old = B.copy()

                    B, allQ = sess.run([predict, Qout], feed_dict={input_NN: S})

                    S_ = S.copy()

                    if B == 0:
                        A = 'Lane1'

                    if B == 1:
                        A = 'Lane2'

                    if B == 2:
                        A = 'Lane3'

                    if B == 3:
                        A = 'Lane4'

                    if (np.random.uniform() > EPSILON):
                        A = np.random.choice(ACTIONS)

                   if (A == 'Lane1' or (np.equal(S[0, 0], 1) and np.equal(S[0, 2], 0) and np.equal(S[0, 4], 0) and np.equal(S[0, 6], 0))
                            or (np.equal(S[0, 0], 1) and np.equal(S[0, 2], 0) and np.equal(S[0, 4], 1) and np.equal(S[0, 6], 0))):
                        if vehicles_waiting_1:
                            # ADD GET POSITION OF 1st VEHICLE
                            traci.vehicle.setStop(vehicles_waiting_1[0], "1i", position1[0], 0, 0)
                            traci.vehicle.setSpeed(vehicles_waiting_1[0], 10)
                            traci.vehicle.setSpeedMode(vehicles_waiting_1[0], 0)
                            wt_time_1 = traci.vehicle.getWaitingTime(vehicles_waiting_1[0])
                            vehicles_waiting_1.pop(0)
                            position1.pop(0)

                    if (A == 'Lane2' or (np.equal(S[0, 0], 0) and np.equal(S[0, 2], 1) and np.equal(S[0, 4], 0) and np.equal(S[0, 6], 0))
                            or (np.equal(S[0, 0], 0) and np.equal(S[0, 2], 1) and np.equal(S[0, 4], 0) and np.equal(S[0, 6], 1))):
                        if vehicles_waiting_2:
                            # ADD GET POSITION OF 1st VEHICLE
                            traci.vehicle.setStop(vehicles_waiting_2[0], "3i", position2[0], 0, 0)
                            traci.vehicle.setSpeed(vehicles_waiting_2[0], 10)
                            traci.vehicle.setSpeedMode(vehicles_waiting_2[0], 0)
                            wt_time_2 = traci.vehicle.getWaitingTime(vehicles_waiting_2[0])
                            vehicles_waiting_2.pop(0)
                            position2.pop(0)

                    if (A == 'Lane3' or (np.equal(S[0, 0], 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 4], 1) and np.equal(S[0, 6], 0))
                            or (np.equal(S[0, 0], 1) and np.equal(S[0, 2], 0) and np.equal(S[0, 4], 1) and np.equal(S[0, 6], 0))):
                        if vehicles_waiting_3:
                            # ADD GET POSITION OF 1st VEHICLE
                            traci.vehicle.setStop(vehicles_waiting_3[0], "2i", position3[0], 0, 0)
                            traci.vehicle.setSpeed(vehicles_waiting_3[0], 10)
                            traci.vehicle.setSpeedMode(vehicles_waiting_3[0], 0)
                            wt_time_3 = traci.vehicle.getWaitingTime(vehicles_waiting_3[0])
                            vehicles_waiting_3.pop(0)
                            position3.pop(0)

                    if (A == 'Lane4' or (np.equal(S[0, 0], 0) and np.equal(S[0, 2], 0) and np.equal(S[0, 4], 0) and np.equal(S[0, 6], 1))
                            or (np.equal(S[0, 0], 0) and np.equal(S[0, 2], 1) and np.equal(S[0, 4], 0) and np.equal(S[0, 6], 1))):
                        if vehicles_waiting_4:
                            # ADD GET POSITION OF 1st VEHICLE
                            traci.vehicle.setStop(vehicles_waiting_4[0], "4i", position4[0], 0, 0)
                            traci.vehicle.setSpeed(vehicles_waiting_4[0], 10)
                            traci.vehicle.setSpeedMode(vehicles_waiting_4[0], 0)
                            wt_time_4 = traci.vehicle.getWaitingTime(vehicles_waiting_4[0])
                            vehicles_waiting_4.pop(0)
                            position4.pop(0)

                    if 495 not in position1:
                        if vehicles_waiting_1:
                            pos1 = 495
                            i1 = 0
                            except1 = 0
                            change_position1 = []
                            for vehicle1 in vehicles_waiting_1:
                                try:
                                    traci.vehicle.setStop(vehicle1, "1i", position1[i1], 0, 0)
                                    traci.vehicle.setStop(vehicle1, "1i", pos1)
                                    change_position1.append(pos1)
                                    pos1 -= 3
                                    i1 += 1
                                except Exception:
                                    except1 = 1
                                    continue
                            if except1:
                                traci.vehicle.remove(vehicles_waiting_1[0], reason=3)
                                if vehicles_waiting_1[0] in managed_vehicles:
                                    managed_vehicles.remove(vehicles_waiting_1[0])
                                    vehicles_in_loop_1.remove(vehicles_waiting_1[0])
                                if vehicles_waiting_1[0] in running_vehicle_ids:
                                    running_vehicle_ids.remove(vehicles_waiting_1[0])
                                vehicles_waiting_1.pop(0)
                                position1.pop(0)
                            else:
                                position1 = change_position1

                    if 495 not in position2:
                        if vehicles_waiting_2:
                            pos2 = 495
                            i2 = 0
                            except2 = 0
                            change_position2 = []
                            for vehicle2 in vehicles_waiting_2:
                                try:
                                    traci.vehicle.setStop(vehicle2, "3i", position2[i2], 0, 0)
                                    traci.vehicle.setStop(vehicle2, "3i", pos2)
                                    change_position2.append(pos2)
                                    pos2 -= 3
                                    i2 += 1
                                except Exception:
                                    except2 = 1
                                    continue
                            if except2:
                                traci.vehicle.remove(vehicles_waiting_2[0], reason=3)
                                if vehicles_waiting_2[0] in managed_vehicles:
                                    managed_vehicles.remove(vehicles_waiting_2[0])
                                    vehicles_in_loop_2.remove(vehicles_waiting_2[0])
                                if vehicles_waiting_2[0] in running_vehicle_ids:
                                    running_vehicle_ids.remove(vehicles_waiting_2[0])
                                vehicles_waiting_2.pop(0)
                                position2.pop(0)
                            else:
                                position2 = change_position2

                    if 495 not in position3:
                        if vehicles_waiting_3:
                            pos3 = 495
                            i3 = 0
                            except3 = 0
                            change_position3 = []
                            for vehicle3 in vehicles_waiting_3:
                                try:
                                    traci.vehicle.setStop(vehicle3, "2i", position3[i3], 0, 0)
                                    traci.vehicle.setStop(vehicle3, "2i", pos3)
                                    change_position3.append(pos3)
                                    pos3 -= 3
                                    i3 += 1
                                except Exception:
                                    except3 = 1
                                    continue
                            if except3:
                                traci.vehicle.remove(vehicles_waiting_3[0], reason=3)
                                if vehicles_waiting_3[0] in managed_vehicles:
                                    managed_vehicles.remove(vehicles_waiting_3[0])
                                    vehicles_in_loop_3.remove(vehicles_waiting_3[0])
                                if vehicles_waiting_3[0] in running_vehicle_ids:
                                    running_vehicle_ids.remove(vehicles_waiting_3[0])
                                vehicles_waiting_3.pop(0)
                                position3.pop(0)
                            else:
                                position3 = change_position3

                    if 495 not in position4:
                        if vehicles_waiting_4:
                            pos4 = 495
                            i4 = 0
                            except4 = 0
                            change_position4 = []
                            for vehicle4 in vehicles_waiting_4:
                                try:
                                    traci.vehicle.setStop(vehicle4, "4i", position4[i4], 0, 0)
                                    traci.vehicle.setStop(vehicle4, "4i", pos4)
                                    change_position4.append(pos4)
                                    pos4 -= 3
                                    i4 += 1
                                except Exception:
                                    except4 = 1
                                    continue
                            if except4:
                                traci.vehicle.remove(vehicles_waiting_4[0], reason=3)
                                if vehicles_waiting_4[0] in managed_vehicles:
                                    managed_vehicles.remove(vehicles_waiting_4[0])
                                    vehicles_in_loop_4.remove(vehicles_waiting_4[0])
                                if vehicles_waiting_4[0] in running_vehicle_ids:
                                    running_vehicle_ids.remove(vehicles_waiting_4[0])
                                vehicles_waiting_4.pop(0)
                                position4.pop(0)
                            else:
                                position4 = change_position4

                    traci.simulationStep()

                    add_vehicles_1 = traci.inductionloop.getLastStepVehicleIDs("e1Detector_1i_0_0")
                    remove_vehicles_1 = traci.inductionloop.getLastStepVehicleIDs("e1Detector_2o_0_5")
                    add_vehicles_3 = traci.inductionloop.getLastStepVehicleIDs("e1Detector_2i_0_2")
                    remove_vehicles_3 = traci.inductionloop.getLastStepVehicleIDs("e1Detector_1o_0_6")
                    add_vehicles_2 = traci.inductionloop.getLastStepVehicleIDs("e1Detector_3i_0_1")
                    remove_vehicles_2 = traci.inductionloop.getLastStepVehicleIDs("e1Detector_4o_0_7")
                    add_vehicles_4 = traci.inductionloop.getLastStepVehicleIDs("e1Detector_4i_0_3")
                    remove_vehicles_4 = traci.inductionloop.getLastStepVehicleIDs("e1Detector_3o_0_4")

                    add_vehicles(add_vehicles_1)
                    add_vehicles(add_vehicles_2)
                    add_vehicles(add_vehicles_3)
                    add_vehicles(add_vehicles_4)

                    remove_vehicles(remove_vehicles_1)
                    remove_vehicles(remove_vehicles_2)
                    remove_vehicles(remove_vehicles_3)
                    remove_vehicles(remove_vehicles_4)

                    for vehicle1 in add_vehicles_1:
                        if vehicle1 not in vehicles_in_loop_1:
                            vehicles_in_loop_1.append(vehicle1)
                        if vehicle1 not in vehicles_waiting_1:
                            vehicles_waiting_1.append(vehicle1)
                            if position1:
                                i = len(position1)
                                tmp_pos1 = position1[i - 1]
                                position1.append(tmp_pos1 - 3)
                                traci.vehicle.setStop(vehicle1, "1i", position1[i])
                            else:
                                for i in range(0, len(vehicles_waiting_1)):
                                    position1.append(495 - 3 * i)
                                    traci.vehicle.setStop(vehicle1, "1i", position1[i])

                    for vehicle3 in add_vehicles_3:
                        if vehicle3 not in vehicles_in_loop_3:
                            vehicles_in_loop_3.append(vehicle3)
                        if vehicle3 not in vehicles_waiting_3:
                            vehicles_waiting_3.append(vehicle3)
                            if position3:
                                j = len(position3)
                                tmp_pos3 = position3[j - 1]
                                position3.append(tmp_pos3 - 3)
                                traci.vehicle.setStop(vehicle3, "2i", position3[j])
                            else:
                                for j in range(0, len(vehicles_waiting_3)):
                                    position3.append(495 - 3 * j)
                                    traci.vehicle.setStop(vehicle3, "2i", position3[j])

                    for vehicle2 in add_vehicles_2:
                        if vehicle2 not in vehicles_in_loop_2:
                            vehicles_in_loop_2.append(vehicle2)
                        if vehicle2 not in vehicles_waiting_2:
                            vehicles_waiting_2.append(vehicle2)
                            if position2:
                                k = len(position2)
                                tmp_pos2 = position2[k - 1]
                                position2.append(tmp_pos2 - 3)
                                traci.vehicle.setStop(vehicle2, "3i", position2[k])
                            else:
                                for k in range(0, len(vehicles_waiting_2)):
                                    position2.append(495 - 3 * k)
                                    traci.vehicle.setStop(vehicle2, "3i", position2[k])

                    for vehicle4 in add_vehicles_4:
                        if vehicle4 not in vehicles_in_loop_4:
                            vehicles_in_loop_4.append(vehicle4)
                        if vehicle4 not in vehicles_waiting_4:
                            vehicles_waiting_4.append(vehicle4)
                            if position4:
                                l = len(position4)
                                tmp_pos4 = position4[l - 1]
                                position4.append(tmp_pos4 - 3)
                                traci.vehicle.setStop(vehicle4, "4i", position4[l])
                            else:
                                for l in range(0, len(vehicles_waiting_4)):
                                    position4.append(495 - 3 * l)
                                    traci.vehicle.setStop(vehicle4, "4i", position4[l])

                    if remove_vehicles_1:
                        if remove_vehicles_1[0] in vehicles_in_loop_1:
                            vehicles_in_loop_1.pop(0)

                    if remove_vehicles_2:
                        if remove_vehicles_2[0] in vehicles_in_loop_2:
                            vehicles_in_loop_2.pop(0)

                    if remove_vehicles_3:
                        if remove_vehicles_3[0] in vehicles_in_loop_3:
                            vehicles_in_loop_3.pop(0)

                    if remove_vehicles_4:
                        if remove_vehicles_4[0] in vehicles_in_loop_4:
                            vehicles_in_loop_4.pop(0)

                    S_ = set_state_space(S_, vehicles_in_loop_1, vehicles_in_loop_2, vehicles_in_loop_3,
                                         vehicles_in_loop_4)
										 
					R = len(remove_vehicles_1) + len(remove_vehicles_2) + len(remove_vehicles_3) + len(remove_vehicles_4)

                    R_ = traci.simulation.getCollidingVehiclesNumber()
                    R_List = traci.simulation.getCollidingVehiclesIDList()

                    for vehicleC in R_List:
                        if vehicleC not in collided_vehicles:
                            collided_vehicles.append(vehicleC)
                            list_is_new = True

                    if R_ and list_is_new == True:
                        R_collision = 1
                        collision_reward = -1
                        R_ = R_ * collision_reward / 2.75

                    total_reward += R + R_
                    step += 1

                    waiting_time += wt_time_1 + wt_time_2 + wt_time_3 + wt_time_4
                    global_step += 1

                    S = S_


            traci.close()
            sys.stdout.flush()

            # empty lists before new episode starts

            managed_vehicles = []

        print("Finish")


def add_vehicles(vehicles):
    global managed_vehicles
    for vehicle in vehicles:
        if not vehicle in managed_vehicles:
            managed_vehicles.append(vehicle)

def set_state_space(S, a, b, c, d):

    if a:
        np.put(S, 0, 1)
        plc1 = traci.vehicle.getLanePosition(a[0])
        np.put(S, 1, round(plc1))
    else:
        np.put(S, 0, 0)
        np.put(S, 1, 0)

    if b:
        np.put(S, 2, 1)
        plc2 = traci.vehicle.getLanePosition(b[0])
        np.put(S, 3, round(plc2))
    else:
        np.put(S, 2, 0)
        np.put(S, 3, 0)

    if c:
        np.put(S, 4, 1)
        plc3 = traci.vehicle.getLanePosition(c[0])
        np.put(S, 5, round(plc3))
    else:
        np.put(S, 4, 0)
        np.put(S, 5, 0)

    if d:
        np.put(S, 6, 1)
        plc4 = traci.vehicle.getLanePosition(d[0])
        np.put(S, 7, round(plc4))
    else:
        np.put(S, 6, 0)
        np.put(S, 7, 0)

    return S


def remove_vehicles(vehicles):
    global managed_vehicles
    for vehicle in vehicles:
        if vehicle in managed_vehicles:
            managed_vehicles.remove(vehicle)


def generate_routefile():
    random.seed(42)  # make tests reproducible
    N = 30000 # number of time steps
    # demand per second from different directions
    traffic_density = [random.randint(-3, 3) for i in range(0, 4)]
    pWE = 1. / (9 - traffic_density[0])   # 1. / 15
    pEW = 1. / (9 - traffic_density[1])  # 1. / 13
    pNS = 1. / (9 - traffic_density[2])   # 1. / 20
    pSN = 1. / (9 - traffic_density[3])   # 1. / 25

    with open("data/cross.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="typeWE" accel="0.4" decel="6.5" sigma="0.5" length="5" minGap="1.5" maxSpeed="10" guiShape="passenger"/>
        <vType id="typeNS" accel="0.4" decel="6.5" sigma="0.5" length="5" minGap="1.5" maxSpeed="10" guiShape="passenger"/>
        <route id="right" edges="51o 1i 2o 52i" />
        <route id="left" edges="52o 2i 1o 51i" />
        <route id ="up" edges="53o 3i 4o 54i" />
        <route id="down" edges="54o 4i 3o 53i" />""", file=routes)
        lastVeh = 0
        vehNr = 0
        for i in range(N):
            if random.uniform(0, 1) < pWE:
                print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i
                continue
            if random.uniform(0, 1) < pEW:
                print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i
                continue
            if random.uniform(0, 1) < pSN:
                print('    <vehicle id="up_%i" type="typeNS" route="up" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i
                continue
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="down_%i" type="typeNS" route="down" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i
                continue
        print("</routes>", file=routes)

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

if __name__ == '__main__':
    playGame()
