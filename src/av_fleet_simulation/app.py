SIMULATION_STEPS = 1


def run(model):

    i = 0
    while i < SIMULATION_STEPS:
        print(f"--- step {i} ---")
        model.step()
        i += 1
