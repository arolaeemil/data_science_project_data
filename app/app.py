from rnn import train
from utils import get_prices, write_prices

# write_prices()

values = get_prices()

SEQ_LEN = 300

model = train(values["NDA-FI"], sequence_length=SEQ_LEN, target_offset=300, epochs=50)

to_predict_from = values["NDA-FI"][-300:]

assert len(to_predict_from) == SEQ_LEN

prediction = model.predict_next(to_predict_from)
print(prediction)
