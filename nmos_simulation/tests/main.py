from test_compare_sims import *
import os


if __name__ == "__main__":
    game = "Pitfall"
    game2rom = {"Pitfall": params.ROMS_PITFALL,
                "DonkeyKong": params.ROMS_DONKEY_KONG,
                "SpaceInvaders": params.ROMS_SPACE_INVADERS,
                }
    iteration_time = 512
    path = "{}/original_3510_512.npy".format(game)

    print("Simulation start!")
    # Do High/Low Voltage single element lesion anaylsis
    # data = single_leision_measure(lesion="High", rom=params.ROMS_SPACE_INVADERS, iteration=511)
    # data = single_leision_measure(lesion="Low", rom=params.ROMS_SPACE_INVADERS, iteration=511)

    # Collect regular simulation data
    data = original_measure(rom=game2rom[game], iteration=iteration_time-1)
    try:
        np.save(path, data)
        print("Simulation end and save file at {}!".format(path))
    except LookupError as e:
        print("Simulation end but save file failed!\n{e}")