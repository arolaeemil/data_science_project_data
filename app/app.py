from rnn import train
from utils import get_prices, write_prices

# write_prices()

values = get_prices()

data = values["NDA-FI"]

print(len(data))

model = train(data, sequence_length=300, target_offset=300, epochs=50)

to_predict_from = data[-300:]
print(to_predict_from)

prediction = model.predict_next(to_predict_from)
print(prediction)


