from rnn import train, train_further, evaluate
from utils import get_prices, write_prices, companies, get_normalized
from matplotlib import pyplot as plt
from app import train_for

# Comparing results with non-normalized input and normalized input data. Apparently current training functions still normalize anyway.

values1 = get_prices()
values2 = get_normalized()

# RAW DATA INPUT AS INPUT

data1 = values1
evaluations1 = []
companylist = companies()

seqlen = 100
offset = 30
epochs = 1
learning_rate = 0.001

model1 = train(values=data1[companylist[0]], sequence_length=seqlen, target_offset=offset, epochs=epochs, learning_rate=learning_rate)

iterations = 3

eval_data_lists1 = [data1[company] for company in companylist[-3:-1] + [companylist[60]]]
print(len(eval_data_lists1))
print(len(eval_data_lists1[0]))

for i in range(iterations):
	print(f"Iteration {i+1}..." + "="*40)
	for company in companylist[1:50]:
		print(f"Training for {company}...")
		try:
			model1, ev = train_for(
				model=model1,
				data=data1[company],
				eval_data_lists=eval_data_lists1,
				sequence_length=seqlen,
				target_offset=offset,
				epochs=epochs,
				learning_rate=learning_rate
			)
			evaluations1.append(ev)
		except ValueError as e:
			print(f"Error training for {company}: {e}")


# NORMALIZED DATA AS INPUT      

data2 = values2
evaluations2 = []

seqlen = 100
offset = 30
epochs = 1
learning_rate = 0.001

model2 = train(values=data2[companylist[0]], sequence_length=seqlen, target_offset=offset, epochs=epochs, learning_rate=learning_rate)

iterations = 3

eval_data_lists2 = [data2[company] for company in companylist[-3:-1] + [companylist[60]]]
print(len(eval_data_lists2))
print(len(eval_data_lists2[0]))


for i in range(iterations):
	print(f"Iteration {i+1}..." + "="*40)
	for company in companylist[1:50]:
		print(f"Training for {company}...")
		try:
			model2, ev = train_for(
				model=model2,
				data=data2[company],
				eval_data_lists=eval_data_lists2,
				sequence_length=seqlen,
				target_offset=offset,
				epochs=epochs,
				learning_rate=learning_rate
			)
			evaluations2.append(ev)
		except ValueError as e:
			print(f"Error training for {company}: {e}")

# plotting
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
axes[0].plot(evaluations1)
axes[0].set_ylabel("Evaluation")
axes[0].set_title("Model 1 Performance")

axes[1].plot(evaluations2)
axes[1].set_ylabel("Evaluation")
axes[1].set_title("Model 2 Performance")

plt.tight_layout()
plt.show()


print(evaluations1)
print(evaluations2)