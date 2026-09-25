SIMULATION_STEPS = 2


def run(model):

    i = 0
    while i < SIMULATION_STEPS:
        print(f"--- step {i} ---")
        model.step()
        i += 1
