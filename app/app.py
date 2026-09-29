from rnn import train

values = [1, 2, 1, 2, 1]
model = train(values, sequence_length=3, target_offset=1)
prediction = model.predict_next(values[-3:])
print(prediction)
