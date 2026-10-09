from rnn import train, train_further, evaluate
from utils import get_prices, write_prices, companies
from matplotlib import pyplot as plt

# write_prices()

values = get_prices()

def train_for(model, data, eval_data_lists, sequence_length, target_offset, epochs, learning_rate):
	model = train_further(
		model=model,
		values=data,
		sequence_length=sequence_length,
		target_offset=target_offset,
		epochs=epochs,
		learning_rate=learning_rate
	)
	this_ev = []
	for eval_data in eval_data_lists:
		if len(eval_data) < sequence_length + target_offset:
			raise ValueError(f"Evaluation data length {len(eval_data)} is less than required {sequence_length + target_offset}")
		ev = evaluate(
			model=model,
			values=eval_data,
			sequence_length=sequence_length,
			target_offset=target_offset
		)
		this_ev.append(ev)
	return model, this_ev

if __name__ == "__main__":
	evaluations = []
	data = values
	companylist = companies()

	seqlen = 100
	offset = 30
	epochs = 1
	learning_rate = 0.001

	model = train(values=data[companylist[0]], sequence_length=seqlen, target_offset=offset, epochs=epochs, learning_rate=learning_rate)

	iterations = 3

	eval_data_lists = [data[company] for company in companylist[-3:-1] + [companylist[60]]]

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
			except ValueError as e:
				print(f"Error training for {company}: {e}")

	plt.plot(evaluations)
	plt.ylabel("Evaluation")
	plt.title("Model Performance")
	plt.show()


	print(evaluations)
