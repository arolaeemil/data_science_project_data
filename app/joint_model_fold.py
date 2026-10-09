from rnn import train, train_further, evaluate
from utils import get_prices, write_prices, companies, get_normalized
from matplotlib import pyplot as plt
from app import train_for
from sklearn.model_selection import KFold

# 10-fold attempt, every time approx 10% is used to evaluation and rest training, takes long time

values = get_prices()
#values = get_normalized()


data = values
evaluations = []
companylist = companies()

seqlen = 100
offset = 30
epochs = 1
learning_rate = 0.001

kf = KFold(n_splits=10, shuffle=True, random_state=42)

iterations = 3
#evaluations = []
fold_evaluations = []

for fold, (train_indices, test_indices) in enumerate(kf.split(companylist)):
    print(f"Fold {fold + 1}/10")

    first_company = companylist[train_indices[0]]
    fold_eval_data_lists = [data[company] for company in [companylist[i] for i in test_indices]if len(data[company]) >= seqlen + offset]
    #print(len(fold_eval_data_lists))
    #print(len(fold_eval_data_lists[0]))
    
	
    
    model = train(
        values=data[first_company],
        sequence_length=seqlen,
        target_offset=offset,
        epochs=epochs,
        learning_rate=learning_rate
    )

    for iteration in range(iterations):
        print(f"Iteration {iteration + 1}/3")

        for index in train_indices[1:]:
            try:
                company = companylist[index]
                #print(len(data[company]))
                model, ev = train_for(
					model=model,
					data=data[company],
					eval_data_lists=fold_eval_data_lists,
					sequence_length=seqlen,
					target_offset=offset,
					epochs=epochs,
					learning_rate=learning_rate
					)
                fold_evaluations.append(ev)
            except ValueError as e:
                    print(f"Error training for {company}: {e}")

plt.plot([
    sum(fold) / len(fold) if fold else float("nan")
    for fold in fold_evaluations
])

plt.xlabel("Fold")
plt.ylabel("Average evaluation")
plt.title("10-fold Cross-Validation")
plt.show()