from rnn import train, train_further, evaluate
from utils import get_prices, write_prices, companies, get_normalized
from matplotlib import pyplot as plt
from app import train_for
import numpy as np

# looping through parameters, the 0.001 learning rate, 1 epochs and 3 iterations seem pretty reasonable.

values = get_prices()
#values = get_normalized()

data = values
evaluations = []
companylist = companies()
evaluations_list = []
parameters_list = []

seqlen = 100
offset = 30
#epochs_list = [1, 2, 5]
epochs_list = [1]
#learning_rates = [0.001, 0.0025, 0.005]
#learning_rates = [0.001, 0.0005]
learning_rates = [0.001]
#iterations_list = [1, 3, 5]
iterations_list = [3]

for epochs in epochs_list:
    for learning_rate in learning_rates:
        for iterations in iterations_list:
            model = train(values=data[companylist[0]], sequence_length=seqlen, target_offset=offset, epochs=epochs, learning_rate=learning_rate)
            eval_data_lists = [data[company] for company in companylist[-3:-1] + [companylist[60]] + [companylist[61]] + [companylist[62]]]
            #print(len(eval_data_lists))
            #print(len(eval_data_lists[0]))
            evaluations = []
            for i in range(iterations):
                print(f"Iteration {i+1}..." + "="*40)
                for company in companylist[1:50]:
                    print(f"Training for {company}...")
                    try:
                        model, ev = train_for(
                            model=model,
                            data=data[company],
                            eval_data_lists=eval_data_lists,
                            sequence_length=seqlen,
                            target_offset=offset,
                            epochs=epochs,
                            learning_rate=learning_rate
                        )
                        evaluations.append(ev)
                        #print(evaluations)
                    except ValueError as e:
                        print(f"Error training for {company}: {e}")
            parameters_list.append([epochs, learning_rate, iterations])
            evaluations_list.append(evaluations)
            print("One parameter setup done")
i = 0
colors = ["red", "blue", "green"]
for evaluation in evaluations_list:
    print(parameters_list[i])
    print(np.mean(evaluation[-1]))
    print("")
    plt.plot(evaluation, color=colors[i])
    i = i + 1

plt.ylabel("Evaluation")
plt.title("Model Performance")
plt.show()
