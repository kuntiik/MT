# import dotenv
import hydra
from omegaconf import DictConfig

# dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="train.yaml")
def main(config: DictConfig):
    from src.training_pipeline import train

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
