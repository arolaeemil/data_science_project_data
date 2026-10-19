
from pathlib import Path
import torch
from utils import companies, get_normalized, get_normalization_data
from utils_open import get_normalized_open, get_normalization_data_open
import numpy as np

# simple test where it is assumed that every iteration 5 best stocks are bought with equal sum and 5 worst are sold. 

def main():

    sequence_length = 100
    target_offset = 30
    limit = 10
    models_directory = Path(__file__).parent / "models" / "test"

    #data = get_normalized()
    #normalization_data = get_normalization_data()
    company_list = companies()

    data = get_normalized_open()
    normalization_data = get_normalization_data_open()

    total_profit = []
    sell_profit = []
    buy_profit = []

    #for cut_by in range(target_offset, limit * target_offset + 1, target_offset):
    for cut_by in range(10*target_offset, 0, -target_offset):
        #model_path = models_directory / (
        #    f"model_cut_by-{cut_by}_seqlen-{sequence_length}"
        #    f"_offset-{target_offset}.pt"
        #)

        model_path = models_directory / (
            f"model_cut_by-{cut_by}_seqlen-{sequence_length}"
            f"_offset-{target_offset}_open.pt"
        )

        if not model_path.exists():
            print(f"\nModel not found: {model_path}")
            continue

        print("\n" + "=" * 100)
        print(f"TRAINING DATA SHORTENED BY {cut_by} STOCK MARKET DAYS, PREDICTION {target_offset} DAYS FORWARD FROM THE NEW END POINT")
        print("=" * 100)

        model = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )
        model.eval()

        predictions = []

        for company in company_list:
            if company not in data or company not in normalization_data:
                continue
            # faulty data for some of the time series
            #if company == "SSABBH" or company == "SSABAH":
                #continue

            values = data[company]
            cutoff = len(values) - cut_by

            # Check that sufficient input and future data exist
            if cutoff < sequence_length:
                continue

            actual_index = cutoff + target_offset - 1

            if actual_index >= len(values):
                continue

            #input_values = values[cutoff - sequence_length:cutoff]
            input_values = values[-(sequence_length + cut_by):-cut_by]
            #print(-(sequence_length + cut_by))
            #print(-cut_by)
            #break

            stats = normalization_data[company]
            mean = stats["mean"]
            std = stats["std"]

            current_price = float(values[cutoff - 1]) * std + mean

            actual_future_price = (float(values[actual_index]) * std + mean)

            if current_price <= 0:
                continue

            # Predict the future price
            with torch.no_grad():
                prediction = model.predict_next(input_values)

            if isinstance(prediction, torch.Tensor):
                prediction = prediction.item()

            predicted_price = float(prediction) * std + mean

            # % changes
            predicted_change = ((predicted_price - current_price) / current_price) * 100

            actual_change = ((actual_future_price - current_price) / current_price) * 100

            error = predicted_change - actual_change

            if predicted_change > 0 and actual_change > 0 or predicted_change < 0 and actual_change < 0:
                result = "PROFIT"
            elif actual_change == 0:
                result = "BREAK EVEN"
            else:
                result = "LOSS"

            predictions.append({
                "company": company,
                "predicted": predicted_change,
                "actual": actual_change,
                "error": error,
                "result": result,
            })

        if not predictions:
            print("No valid predictions.")
            continue

        # Print a list of predictions
        def print_results(title, results):
            print(f"\n{title}")
            print(
                f"{'Company':<12}"
                f"{'Predicted':>12}"
                f"{'Actual':>12}"
                f"{'Error (pp)':>14}"
                f"{'Result':>14}"
            )
            print("-" * 64)

            for p in results:
                print(
                    f"{p['company']:<12}"
                    f"{p['predicted']:>+11.2f}%"
                    f"{p['actual']:>+11.2f}%"
                    f"{p['error']:>+13.2f}"
                    f"{p['result']:>14}"
                )

        sorted_predictions = sorted(predictions, key=lambda p: p["predicted"])

        print_results("5 LOWEST PREDICTED RETURNS (SELL)", sorted_predictions[:5])

        print_results("5 HIGHEST PREDICTED RETURNS (BUY)",sorted_predictions[-5:][::-1])

        # Buy the 5 companies with the highest predicted returns
        best_5 = sorted_predictions[-5:]

        # Short-sell the 5 companies with the lowest predicted returns
        worst_5 = sorted_predictions[:5]

        # Average return from buying the best 5
        best_return = sum(p["actual"] for p in best_5) / len(best_5)

        # Average return from short-selling the worst 5
        # A falling share price generates a positive short return
        worst_return = -sum(p["actual"] for p in worst_5) / len(worst_5)

        # Assume equal capital allocated to the long and short sides
        strategy_return = (best_return + worst_return) / 2

        buy_profit.append(best_return)
        sell_profit.append(worst_return)
        total_profit.append(strategy_return)

        # Overall statistics
        mean_error = sum(p["error"] for p in predictions) / len(predictions)

        profitable = sum(p["actual"] > 0 for p in predictions
                         )
        losing = sum(p["actual"] < 0 for p in predictions
                     )
        break_even = sum(p["actual"] == 0 for p in predictions)

        print("\nOVERALL RESULTS")
        print(f"Companies evaluated: {len(predictions)}")
        print(f"Profitable stocks:   {profitable}")
        print(f"Losing stocks:       {losing}")
        print(f"Break even:          {break_even}")
        print(f"Mean signed error:   {mean_error:+.2f} percentage points")

        print("\nSTRATEGY EFFECTIVENESS")
        print(f"Buying best 5 average return:  {best_return:+.2f}%")
        print(f"Selling worst 5 average return: {worst_return:+.2f}%")
        print(f"Combined strategy return:      {strategy_return:+.2f}%")

    print("\nPrediction complete.")

    total_profit_mean = np.mean(total_profit)
    buy_profit_mean = np.mean(buy_profit)
    sell_profit_mean = np.mean(sell_profit)
    print(f"Mean total profit per time period: {total_profit_mean:+.2f}%")
    print(f"Mean buying profit per time period: {buy_profit_mean:+.2f}%")
    print(f"Mean selling profit per time period: {sell_profit_mean:+.2f}%")

if __name__ == "__main__":
    main()
