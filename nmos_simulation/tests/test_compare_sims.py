import numpy as np
from nose.tools import *

import sim2600
from sim2600 import sim2600Console
from sim2600 import params, sim6502, simTIA
import sim2600.sim6502


def compare_sims(s1func, s2func, ITERS=100):
    for rom in [params.ROMS_DONKEY_KONG, params.ROMS_SPACE_INVADERS,
                params.ROMS_PITFALL]:

        s1 = s1func(rom)
        s2 = s2func(rom)

        s1_init_state = s1.sim6507.getWiresState()  # # getWireState()
        s2_init_state = s2.sim6507.getWiresState()  # # getWireState()

        np.testing.assert_array_equal(s1_init_state, s2_init_state)

        for i in range(ITERS):
            s1.advanceOneHalfClock()
            s2.advanceOneHalfClock()
            s1_state = s1.sim6507.getWiresState()  # # getWireState()
            s2_state = s2.sim6507.getWiresState()  # # getWireState()

            # get transistor state
            t1_state = s1.sim6507.getTransistorState()
            t2_state = s2.sim6507.getTransistorState()
            print(t1_state.shape)
            np.testing.assert_array_equal(s1_state, s2_state)

            s1_tia_state = s1.simTIA.getWiresState()  # # getWireState()
            s2_tia_state = s2.simTIA.getWiresState()  # # getWireState()

            np.testing.assert_array_equal(s1_tia_state, s2_tia_state)


def test_compare_simple_simple():
    """
    Just compare our default simulator agaginst
    itself
    """
    s1 = lambda x: sim2600Console.Sim2600Console(x)

    compare_sims(s1, s1)


def test_compare_list_sets():
    """
    Just compare our default simulator agaginst
    itself
    """
    s1 = lambda x: sim2600Console.Sim2600Console(x, sim6502.Sim6502)
    s2 = lambda x: sim2600Console.Sim2600Console(x, sim6502.Sim6502Sets)

    compare_sims(s1, s2)


def test_compare_list_mine():
    """
    Just compare our default simulator agaginst
    itself
    """
    s1 = lambda x: sim2600Console.Sim2600Console(x, sim6502.Sim6502)
    s2 = lambda x: sim2600Console.Sim2600Console(x, sim6502.MySim6502)

    compare_sims(s1, s2)


def test_compare_list_mine_tia():
    """
    Just compare our default simulator agaginst
    itself
    """
    s1 = lambda x: sim2600Console.Sim2600Console(x)
    s2 = lambda x: sim2600Console.Sim2600Console(x, simTIAfactory=simTIA.MySimTIA)

    compare_sims(s1, s2, ITERS=40)


def test_compare_both():
    """
    Just compare our default simulator agaginst
    itself
    """
    s1 = lambda x: sim2600Console.Sim2600Console(x)
    s2 = lambda x: sim2600Console.Sim2600Console(x, simTIAfactory=simTIA.MySimTIA,
                                                 sim6502factory=sim6502.MySim6502)

    compare_sims(s1, s2, ITERS=400)

def transistor_record(lesion=None, rom=params.ROMS_DONKEY_KONG, iteration=1000, tidx=-1):
    """
    Record the transistor state for the given number of iterations
    """
    if lesion is None:
        # Regular simulation and not lesioning
        tidx = -1
        s1 = lambda x: sim2600Console.Sim2600Console(x, simTIAfactory=simTIA.MySimTIA,
                                                     sim6502factory=sim6502.MySim6502,
                                                     tidx_lesion=tidx)
    else:
        # Simulation with lesioning a transistor
        s1 = lambda x: sim2600Console.Sim2600Console(x, simTIAfactory=simTIA.MySimTIA,
                                                     sim6502factory=sim6502.MySim6502,
                                                     lesion=lesion,
                                                     tidx_lesion=tidx)


    s1 = s1(rom)
    t1_init_state = s1.sim6507.getTransistorState()

    t_orgs = t1_init_state.reshape((-1, 1))

    # np.testing.assert_array_equal(t1_init_state, t2_init_state)
    print("-" * 50)
    for i in range(iteration):
        s1.advanceOneHalfClock()

        # get transistor state
        t1_state = s1.sim6507.getTransistorState()

        # Calculate Difference
        t_org = t1_state.reshape((-1, 1))
        t_orgs = np.concatenate((t_orgs, t_org), axis=1)

    return t_orgs


def transistors_differ(lesion=None, iteration=1000):
    rom = params.ROMS_DONKEY_KONG
    # tmp = np.load("original_3510_400.npy")
    # vcc_transistors = [idx for idx in range(3510) if tmp[0, idx, :].all() == 1]

    for tidx in range(3510):
        # if tidx not in vcc_transistors:
        #     continue

        s1 = lambda x: sim2600Console.Sim2600Console(x, simTIAfactory=simTIA.MySimTIA,
                                                     sim6502factory=sim6502.MySim6502,
                                                     lesion=lesion,
                                                     tidx_lesion=tidx)
        s2 = lambda x: sim2600Console.Sim2600Console(x, simTIAfactory=simTIA.MySimTIA,
                                                     sim6502factory=sim6502.MySim6502)

        s1 = s1(rom)
        s2 = s2(rom)

        t1_init_state = s1.sim6507.getTransistorState()
        t2_init_state = s2.sim6507.getTransistorState()

        t_orgs = t2_init_state.reshape((-1, 1))
        t_diff = t1_init_state.reshape((-1, 1))

        # np.testing.assert_array_equal(t1_init_state, t2_init_state)
        print("-" * 50)
        for i in range(iteration):
            try:
                s1.advanceOneHalfClock()
            except:
                pass
            s2.advanceOneHalfClock()

            # get transistor state
            ## lesion
            t1_state = s1.sim6507.getTransistorState()
            ## non-lesion
            t2_state = s2.sim6507.getTransistorState()

            # Calculate Difference
            t_org = t2_state.reshape((-1, 1))
            t_diff = t1_state.reshape((-1, 1))
            t_orgs = np.concatenate((t_orgs, t_org), axis=1)
            t_diffs = np.concatenate((t_diffs, t_diff), axis=1)

        if tidx == 0:
            data0 = np.expand_dims(t_orgs.copy(), axis=0)
            data = np.expand_dims(t_diffs.copy(), axis=0)
        else:
            data0 = np.concatenate((data0, np.expand_dims(t_orgs, axis=0)), axis=0)
            data = np.concatenate((data, np.expand_dims(t_diffs, axis=0)), axis=0)

    return data0, data


def original_measure(rom=params.ROMS_DONKEY_KONG, iteration=1000):
    t_org = transistor_record(rom, iteration=iteration)
    print(t_org.shape)
    return t_org


def single_leision_measure(lesion="High", rom=params.ROMS_DONKEY_KONG, iteration=1000):
    for tidx in range(3510):
        t_org = transistor_record(lesion, rom, iteration=iteration, tidx=tidx)
        if tidx == 0:
            data0 = np.expand_dims(t_org.copy(), axis=0)
        else:
            data0 = np.concatenate((data0, np.expand_dims(t_org, axis=0)), axis=0)
    return data0


def causality_measure(iteration=1000):
    # Measure causality effect by transistors difference when force HIGH voltage lesion
    t_org, t_causal = transistors_differ(lesion="High", iteration=iteration)
    print(t_causal.shape)
    np.save("original_3510_512.npy", t_org[0])
    np.save("high_3510_512.npy", t_causal)
