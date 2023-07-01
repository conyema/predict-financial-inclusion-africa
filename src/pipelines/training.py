from src.components.model_training import ModelTrainer
from src.logger import logging


def main():
    logging.info(">>>>> model training/development started <<<<<")

    # Instantiate the processor
    trainer = ModelTrainer()

    # Get clean and endode/tranform it
    trainer.initiate_training()

    # Train candidate models
    # trainer.train(save_best_model=True, save_report=True, show_plots=True)
    trainer.train(save_best_model=True, save_report=True)

    # Train optimized models without sampling
    # trainer.optimized_train(
    #     save_best_model=True, save_report=True, report_title='optimized_not_sampled')

    # Correct class imbalance
    trainer.sample_imbalanced_data()

    # Train optimized models with sampling
    trainer.optimized_train(
        save_best_model=True, save_report=True, report_title='optimized_and_sampled')
    

    logging.info(">>>>> model training/development completed <<<<<")



if __name__ == "__main__":
    main()
