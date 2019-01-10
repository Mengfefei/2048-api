import numpy as np
from keras.models import load_model
import keras

from .expectimax import board_to_move 
search_func = board_to_move

model = load_model('/home/jack97/2048-api-master/model.h5')

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        X = np.zeros((0,4,4))
        Y = np.zeros((0))
        while (n_iter < max_iter) and (not self.game.end):

            data_X = self.game.board
            data_Y = search_func(self.game.board)
            X = np.concatenate((X,data_X.reshape(1,4,4)), axis=0)
            Y = np.concatenate((Y,np.asarray(data_Y).reshape(1)), axis=0)

            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)
        return X,Y

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction

OUT_SHAPE = (4,4)
CAND = 12
map_table = {2**i: i for i in range(1, CAND)}
map_table[0] = 0

def grid_ohe(arr):
    ret = np.zeros(shape=OUT_SHAPE + (CAND,))
    for r in range(OUT_SHAPE[0]):
        for c in range(OUT_SHAPE[1]):
            ret[r,c,map_table[arr[r,c]]] = 1
    return ret


class MyAgent(Agent):

    def step(self):
        board_1 = grid_ohe(self.game.board).reshape(1,4,4,CAND)
        direction = np.argmax(model.predict(board_1))
        return direction

class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction


