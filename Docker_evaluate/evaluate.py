import numpy as np
from tensorflow.keras.models import load_model
import argparse

import joblib

def evaluate_model(x_test, y_test, model_path):
    x_test = np.load(x_test)
    y_test = np.load(y_test)

    model = joblib.load(model_path)
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test lolss: ', score[0])
    print('Test accuracy: ', score[1])

    with open('output.txt', 'a') as f:
        f.write(str(score))
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_test')
    parser.add_argument('--y_test')
    parser.add_argument('--model')
    args = parser.parse_args()
    evaluate_model(args.x_test, args.y_test, args.model)

    #x_test = np.load(args.x_test)
    #y_test = np.load(args.y_test)
    #model = load_model(args.model)

    #score = evaluate_model(model, x_test, y_test)
    #print('Test loss: ', score[0])
    #print('Test accuracy: ', score[1])
    
