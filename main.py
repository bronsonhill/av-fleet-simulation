from src.av_fleet_simulation.model import MultiFleetTrafficModel, ModelParameters

SIMULATION_STEPS = 5

def main():
    model = MultiFleetTrafficModel(ModelParameters(1))

    i = 0 
    while i < SIMULATION_STEPS:
        model.step()
        i += 1
    
    ...

if __name__ == "__main__":
    main()