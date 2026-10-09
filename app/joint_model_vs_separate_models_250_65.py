from rnn import train, train_further, evaluate
from utils import get_prices, write_prices, companies, get_normalized
from matplotlib import pyplot as plt
from app import train_for
import numpy as np

# compare models with seqlen of 250 and target offset of 65. Corresponds pretty much to year in stockmarket and one quarter.

#JOINT MODEL

values = get_prices()
#values = get_normalized()

data = values
evaluations1 = []
companylist = companies()

seqlen = 250
offset = 65
epochs = 1
learning_rate = 0.001

model = train(values=data[companylist[0]], sequence_length=seqlen, target_offset=offset, epochs=epochs, learning_rate=learning_rate)

iterations = 3

for i in range(iterations):
	print(f"Iteration {i+1}..." + "="*40)
	for company in companylist[1:]:
		print(f"Training for {company}...")
		try:
			model = train_further(
				model=model,
				values=data[company],
				sequence_length=seqlen,
				target_offset=offset,
				epochs=epochs,
				learning_rate=learning_rate
			)
		except ValueError as e:
			print(f"Error training for {company}: {e}")

#print(model)
i = 0
joint_model_errors = []
for company in companylist:
    try:
        to_predict_from = data[companylist[i]][-315:-65]
        prediction = model.predict_next(to_predict_from)
        print("company:" + str(company))
        print("predicted: " + str(prediction))
        actual_value = data[companylist[i]][-1]
        print("actual: " + str(actual_value))
        joint_model_errors.append(abs(prediction - actual_value)/actual_value)
    except:
        pass
    i += 1

print("mean relative error (abs) for joint model:")
print(np.mean(joint_model_errors))

# SEPARATE MODEL

SEQ_LEN = 250

separate_model_errors = []
i = 0
while i < len(companylist):
#while i < 5:
    # leave last 250 out so training and test data won't mix, don't crash if some company has only short history
    try:
        model = train(data[companylist[i]][:-65], sequence_length=SEQ_LEN, target_offset=65, epochs=10)
    except:
         i += 1
         continue

    # input is last 250 values before the test data
    to_predict_from = data[companylist[i]][-315:-65:]
    # check for proper length
    assert len(to_predict_from) == SEQ_LEN
    # make prediction, should be the last available datapoint from the current dataset
    prediction = model.predict_next(to_predict_from)

    # printing
    print(companylist[i])
    print("predicted: " + str(prediction))
    actual_value = data[companylist[i]][-1]
    print("actual: " + str(actual_value))
    difference = abs(prediction - actual_value)
    separate_model_errors.append(difference/actual_value)
    i += 1

# listing relative errors made
print("mean relative error (abs) for separate model:")
print(np.mean(separate_model_errors))
print("")
# prints


print("mean relative error (abs) for joint model:")
print(np.mean(joint_model_errors))
print("mean relative error (abs) for separate model:")
print(np.mean(separate_model_errors))