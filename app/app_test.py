from rnn import train, train_further, evaluate
from utils import companies, get_normalized
from matplotlib import pyplot as plt

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

def load_data():
	return get_normalized(), companies()


def initialize_model(data, companylist, sequence_length, target_offset, epochs, learning_rate):
	return train(
		values=data[companylist[0]],
		sequence_length=sequence_length,
		target_offset=target_offset,
		epochs=epochs,
		learning_rate=learning_rate
	)


def select_companies(companylist, k, which_k):
	if k < 2:
		raise ValueError("k must be at least 2 so there are training and evaluation parts")
	if k > len(companylist):
		raise ValueError("k cannot exceed the number of companies")
	if which_k < 1 or which_k > k:
		raise ValueError("which_k must be between 1 and k")

	base_size, extra_companies = divmod(len(companylist), k)
	parts = []
	start = 0
	for part_number in range(k):
		part_size = base_size + (part_number < extra_companies)
		parts.append(companylist[start:start + part_size])
		start += part_size

	evaluation_companies = parts[which_k - 1]
	training_companies = [
		company
		for part_number, part in enumerate(parts, start=1)
		if part_number != which_k
		for company in part
	]
	return training_companies, evaluation_companies


def get_evaluation_data(data, evaluators):
	eval_data_lists = [data[company] for company in evaluators]
	print(f"Using {evaluators} for evaluation")
	return eval_data_lists


def train_on_companies(model, data, companylist, evaluators, eval_data_lists, iterations, sequence_length, target_offset, epochs, learning_rate):
	evaluations = []
	for iteration in range(iterations):
		print(f"Iteration {iteration + 1}..." + "=" * 40)
		for company in companylist:
			if company in evaluators:
				raise ValueError("company in evaluators")
			print(f"Training for {company}...")
			try:
				model, evaluation = train_for(
					model=model,
					data=data[company],
					eval_data_lists=eval_data_lists,
					sequence_length=sequence_length,
					target_offset=target_offset,
					epochs=epochs,
					learning_rate=learning_rate
				)
				evaluations.append(evaluation)
			except ValueError as error:
				print(f"Error training for {company}: {error}")
	return evaluations


def plot_evaluations(evaluations):
	plt.plot(evaluations)
	plt.ylabel("Evaluation")
	plt.title("Error on Evaluation Data")
	plt.show()


def main():
	data, companylist = load_data()
	sequence_length = 100
	target_offset = 30
	epochs = 1
	learning_rate = 0.001
	iterations = 3
	k = 5
	which_k = 5

	training_companies, evaluators = select_companies(companylist, k, which_k)
	model = initialize_model(
		data, training_companies, sequence_length, target_offset, epochs, learning_rate
	)
	eval_data_lists = get_evaluation_data(data, evaluators)
	evaluations = train_on_companies(
		model,
		data,
		training_companies[1:],
		evaluators,
		eval_data_lists,
		iterations,
		sequence_length,
		target_offset,
		epochs,
		learning_rate
	)
	plot_evaluations(evaluations)
	print(evaluations)


if __name__ == "__main__":
	main()
