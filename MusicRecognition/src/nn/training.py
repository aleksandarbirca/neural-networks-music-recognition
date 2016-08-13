from keras.models import Sequential
from keras.layers.core import Dense

def compile_model():
    model = Sequential()
    model.add(Dense(30, init='uniform', activation='sigmoid', input_dim=13))
    model.add(Dense(10, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adagrad')

    json_model = model.to_json()
    open('..\..\data\model.json', 'w').write(json_model)
    print '\nNetwork created successfully and network model saved as file model.json.'



if __name__ == "__main__":
    compile_model()