# import dotenv
import hydra
from omegaconf import DictConfig

# dotenv.load_dotenv(override=True)


@hydra.main(config_path="mt/configs/", config_name="train.yaml")
def main(config: DictConfig):
    from mt.testing_pipeline import test

    # Train model
    return test(config)


if __name__ == "__main__":
    main()
