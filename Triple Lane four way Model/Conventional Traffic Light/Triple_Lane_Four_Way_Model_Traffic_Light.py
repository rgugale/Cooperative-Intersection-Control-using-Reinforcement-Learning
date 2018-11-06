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


MAX_EPISODES = 30
MAX_STEPS = 3000
step = 0

np.random.seed(1337)

global managed_vehicles
managed_vehicles = []

def playGame():

    step = 0

    global managed_vehicles
    managed_vehicles = []

    temp_wt_time = 0
    temp_co2_emission = 0

    # SUMO STUFF

    options = get_options()

    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')


    for episode in range(MAX_EPISODES):

        wt_time = 0
        co2_emission = 0

        generate_routefile()

        traci.start([sumoBinary, "-c", "data/cross.sumocfg",
                     "--tripinfo-output", "tripinfo.xml", "--additional-files",
                     "data/cross.additionals.xml", "--collision.check-junctions",
                     "--collision.action", "warn", "--step-length", "1",
                     "--error-log", "error.txt"])

        for step in range(MAX_STEPS):

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

            if managed_vehicles:
                for vehicle in managed_vehicles:
                    temp_wt_time = traci.vehicle.getWaitingTime(vehicle)
                    temp_co2_emission = traci.vehicle.getCO2Emission(vehicle)

                    wt_time += temp_wt_time
                    co2_emission += temp_co2_emission


            step += 1

        traci.close()
        sys.stdout.flush()


        managed_vehicles = []

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
    N = 100000  # number of time steps
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




