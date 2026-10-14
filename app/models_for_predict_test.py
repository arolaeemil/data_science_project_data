from pathlib import Path
import torch
from rnn import train, train_further, evaluate
from utils import companies, get_normalized
from matplotlib import pyplot as plt

# Note: normalizations are made with whole datasets, so this might affect results somewhat.

def main(save_model=True):
# Parameters
    sequence_length = 100
    target_offset = 30
    epochs = 1
    learning_rate = 0.001
    iterations = 1
    i = target_offset
    limit = 10
    data = get_normalized()
    companylist = companies()

    while i <= limit*target_offset:
        minimum_length = sequence_length + target_offset
        valid_companies = []

        for company in companylist:
            data[company] = data[company][:-target_offset]

        for company in companylist:
            if len(data[company]) >= minimum_length:
                valid_companies.append(company)
            else:
                print(f"Skipping {company}: not enough data")

        if not valid_companies:
            raise ValueError("No companies contain enough data")

        print("=" * 80)
        print("TRAINING COMPANIES:")
        print(", ".join(valid_companies))
        print("=" * 80)

        first_company = valid_companies[0]

        model = train(
            values=data[first_company],
            sequence_length=sequence_length,
            target_offset=target_offset,
            epochs=epochs,
            learning_rate=learning_rate
        )

        for iteration in range(iterations):
            print(f"\nIteration {iteration + 1}/{iterations}")

            for company in valid_companies[1:]:
                print(f"Training for {company}...")

                model = train_further(
                    model=model,
                    values=data[company],
                    sequence_length=sequence_length,
                    target_offset=target_offset,
                    epochs=epochs,
                    learning_rate=learning_rate
                )

                loss = evaluate(
                    model=model,
                    values=data[company],
                    sequence_length=sequence_length,
                    target_offset=target_offset
                )

                print(f"Loss: {loss}")

        if save_model:
            models_directory = Path(__file__).parent / "models/test"
            models_directory.mkdir(exist_ok=True)
            model_path = models_directory / f"model_cut_by-{i}_seqlen-{sequence_length}_offset-{target_offset}.pt"
            torch.save(model, model_path)

            print(f"\nSaved model to {model_path}")

        print("\nTraining complete.")
        i = i + target_offset


if __name__ == "__main__":
    main(save_model=True)
