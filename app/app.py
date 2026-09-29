from rnn import train
from utils import get_prices


values = get_prices()

data = values["NDA-FI"]

model = train(data, sequence_length=3, target_offset=1)
prediction = model.predict_next(data[-3:])
print(prediction)


