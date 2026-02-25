import random
import numpy as np  # Dla macierzy, łatwo na torch

class Game2048Env:
    def __init__(self):
        self.board = self.new_game()
        self.done = False
        self.score = 0

    def new_game(self):
        board = np.zeros((4, 4), dtype=int)
        self.add_random_tile(board)
        self.add_random_tile(board)
        return board

    def add_random_tile(self, board):
        empty_cells = np.argwhere(board == 0)
        if len(empty_cells) > 0:
            i, j = random.choice(empty_cells)
            board[i, j] = random.choice([2, 4])

    # Ruch: Zwraca new_board, reward, changed (dla invalid)
    def move(self, board, direction):  # direction: 0-up, 1-down, 2-left, 3-right
        original_board = board.copy()
        if direction == 0:  # Up
            board = np.rot90(board, 1)  # Transpose + reverse rows (efficient)
            board = self._move_left(board)
            board = np.rot90(board, 3)
        elif direction == 1:  # Down
            board = np.rot90(board, 3)
            board = self._move_left(board)
            board = np.rot90(board, 1)
        elif direction == 2:  # Left
            board = self._move_left(board)
        elif direction == 3:  # Right
            board = np.fliplr(board)
            board = self._move_left(board)
            board = np.fliplr(board)
        
        changed = not np.array_equal(board, original_board)
        reward = 0
        if changed:
            # Reward za merge: Sumuj podwojone wartości (log2 dla skalowania)
            merges = np.sum(board[board > 0] // 2)  # Przybliżone, ale light
            reward += merges * np.log2(merges + 1) if merges > 0 else 0  # Skala log
            if np.max(board) >= 2048:  # Bonus za cel
                reward += 1000
            self.add_random_tile(board)
        else:
            reward -= 1  # Kara za invalid ruch
        
        return board, reward, changed

    def _move_left(self, board):
        new_board = np.zeros_like(board)
        for i in range(4):
            row = board[i][board[i] != 0]
            j = 0
            for k in range(1, len(row)):
                if row[j] == row[k]:
                    row[j] *= 2
                    row = np.delete(row, k)
                else:
                    j = k
            new_board[i, :len(row)] = row
        return new_board

    def is_game_over(self, board):
        if np.any(board == 0):
            return False
        for dir in range(4):
            _, _, changed = self.move(board.copy(), dir)
            if changed:
                return False
        return True

    # Step dla RL: action 0-3, zwraca next_state, reward, done
    def step(self, action):
        self.board, reward, changed = self.move(self.board, action)
        self.done = self.is_game_over(self.board)
        self.score += reward
        return self.board.copy(), reward, self.done

    def reset(self):
        self.board = self.new_game()
        self.done = False
        self.score = 0
        return self.board.copy()

    # Render (opcjonalny, dla debug – bez dla szybkości)
    def render(self):
        print(self.board)

# Test: Symulacja epizodu dla random AI
def simulate_episode(env):
    state = env.reset()
    total_reward = 0
    steps = 0
    while not env.done:
        action = random.randint(0, 3)  # Random policy
        next_state, reward, done = env.step(action)
        total_reward += reward
        steps += 1
    return total_reward, steps, env.board

if __name__ == "__main__":
    env = Game2048Env()
    # Dla człowieka: Symuluj inputy, ale usuń dla pure RL
    # env.render()
    # while not env.done:
    #     dir_map = {'w': 0, 's': 1, 'a': 2, 'd': 3, 'q': -1}
    #     inp = input("Ruch (w/s/a/d/q): ").lower()
    #     if inp == 'q': break
    #     action = dir_map.get(inp, -1)
    #     if action != -1:
    #         _, reward, _ = env.step(action)
    #         env.render()
    # print("Koniec! Score:", env.score)
    
    # Dla AI test: Uruchom symulację
    reward, steps, final_board = simulate_episode(env)
    print("Symulowany epizod: Reward", reward, "Steps", steps)
    print(final_board)