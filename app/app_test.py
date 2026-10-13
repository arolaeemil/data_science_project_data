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
	training_loss_part = 0 if which_k != 1 else 1
	training_loss_companies = parts[training_loss_part]
	training_companies = [
		company
		for part_number, part in enumerate(parts, start=1)
		if part_number != which_k
		for company in part
	]
	return training_companies, evaluation_companies, training_loss_companies


def filter_companies_by_length(data, companylist, sequence_length, target_offset, label):
	minimum_length = sequence_length + target_offset
	valid_companies = []
	for company in companylist:
		if len(data[company]) < minimum_length:
			print(
				f"Removing {company} from {label} companies: "
				f"needs at least {minimum_length} values"
			)
			continue
		valid_companies.append(company)
	return valid_companies


def get_evaluation_data(data, evaluators, sequence_length, target_offset):
	valid_evaluators = filter_companies_by_length(
		data, evaluators, sequence_length, target_offset, "evaluation"
	)
	if not valid_evaluators:
		raise ValueError("No evaluation companies contain enough values")
	eval_data_lists = [data[company] for company in valid_evaluators]
	return valid_evaluators, eval_data_lists


def get_training_loss(model, data, training_loss_companies, sequence_length, target_offset):
	losses = []
	for company in training_loss_companies:
		losses.append(
			evaluate(
				model=model,
				values=data[company],
				sequence_length=sequence_length,
				target_offset=target_offset
			)
		)
	if not losses:
		raise ValueError("No training loss companies contain enough values")
	return sum(losses) / len(losses)


def train_on_companies(model, data, companylist, evaluators, eval_data_lists, training_loss_companies, iterations, sequence_length, target_offset, epochs, learning_rate):
	evaluation_losses = []
	training_losses = []
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
				evaluation_losses.append(sum(evaluation) / len(evaluation))
				training_loss = get_training_loss(
					model, data, training_loss_companies, sequence_length, target_offset
				)
				training_losses.append(training_loss)
				print(f"Training loss: {training_loss:.6f}")
			except ValueError as error:
				print(f"Error training for {company}: {error}")
	return evaluation_losses, training_losses


def plot_evaluations(evaluations, training_losses):
	plt.plot(evaluations, label="Evaluation loss")
	plt.plot(training_losses, label="Training loss")
	plt.ylabel("Mean squared error")
	plt.title("Training and evaluation loss")
	plt.legend()
	plt.show()


def main():
	data, companylist = load_data()
	sequence_length = 100
	target_offset = 30
	epochs = 1
	learning_rate = 0.001
	iterations = 1
	k = 2
	which_k = 2

	training_companies, evaluators, training_loss_companies = select_companies(
		companylist, k, which_k
	)
	training_loss_companies = filter_companies_by_length(
		data, training_loss_companies, sequence_length, target_offset, "training loss"
	)
	print("=" * 80)
	print("EVALUATORS")
	print(evaluators)
	print("=" * 80)
	print("TRAINING COMPANIES")
	print(training_companies)
	print("=" * 80)
	print("TRAINING LOSS COMPANIES")
	print(training_loss_companies)
	print("=" * 80)
	
	model = initialize_model(
		data, training_companies, sequence_length, target_offset, epochs, learning_rate
	)
	evaluators, eval_data_lists = get_evaluation_data(
		data, evaluators, sequence_length, target_offset
	)
	evaluation_losses, training_losses = train_on_companies(
		model,
		data,
		training_companies[1:],
		evaluators,
		eval_data_lists,
		training_loss_companies,
		iterations,
		sequence_length,
		target_offset,
		epochs,
		learning_rate
	)
	plot_evaluations(evaluation_losses, training_losses)
	print(evaluation_losses)


if __name__ == "__main__":
	main()
