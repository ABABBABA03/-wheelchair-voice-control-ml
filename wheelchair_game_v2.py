####################################################################
#                                                                  #
#                      Wheelchair Game                             #
#                                                                  #
#   control a wheelchair through a maze using voice commands.      #
#   Make the blue arrow reach the green square for winning.        #
#   WASD + space keys available for controlling the wheelchair     #
#   simultaneously with the voice commands.                        #
#                                                                  #
#   Libraries needed: pygame, sounddevice, numpy                   #
#                                                                  #
#     Developed by Marco Simoes (msimoes@dei.uc.pt) - DEI 2024     #
#                                                                  #
####################################################################


import pygame
import random
import math
import sounddevice as sd
import numpy as np
import threading
import time
import matplotlib.pyplot as plt

# init pygame
pygame.init()

# define constants
SCREEN_WIDTH, SCREEN_HEIGHT = 855, 675
CELL_SIZE = 15  # cell size
CELL_EXPANSION = 3  # number of cells within each cell
FPS = 10
MAZE_ROWS = SCREEN_HEIGHT // CELL_SIZE // CELL_EXPANSION  # number of rows
MAZE_COLS = SCREEN_WIDTH // CELL_SIZE // CELL_EXPANSION # number of cols
INITIAL_SCORE = 0  # counter of crashes

# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


# game screen set up
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Wheelchair Labyrinth")

last_classification_time = time.time()


class Maze:
    ''' Maze class. Sets up and draws the maze.'''
    
    def __init__(self):
        # base grid
        self.base_grid = [[1 for _ in range(MAZE_COLS)] for _ in range(MAZE_ROWS)]
        # expanded grid
        self.expanded_grid = [[1 for _ in range(MAZE_COLS * CELL_EXPANSION)] for _ in range(MAZE_ROWS * CELL_EXPANSION)]
        
        self.generate_maze()
        self.expand_maze()
        
        # place the target in a randow white cell
        self.target = self.get_random_white_cell()

    def generate_maze(self):
        # simple DFS algorithm to build the maze
        
        stack = [(1, 1)]
        self.base_grid[1][1] = 0

        while stack:
            row, col = stack[-1]
            neighbors = []

            if row > 1 and self.base_grid[row - 2][col] == 1:
                neighbors.append((row - 2, col))
            if row < MAZE_ROWS - 2 and self.base_grid[row + 2][col] == 1:
                neighbors.append((row + 2, col))
            if col > 1 and self.base_grid[row][col - 2] == 1:
                neighbors.append((row, col - 2))
            if col < MAZE_COLS - 2 and self.base_grid[row][col + 2] == 1:
                neighbors.append((row, col + 2))

            if neighbors:
                next_row, next_col = random.choice(neighbors)
                self.base_grid[next_row][next_col] = 0
                self.base_grid[(next_row + row) // 2][(next_col + col) // 2] = 0
                stack.append((next_row, next_col))
            else:
                stack.pop()

    def expand_maze(self):
        # expand rows so each place is a CELL_EXPANSION by CELL_EXPANSION mini-grid of places
        for row in range(MAZE_ROWS):
            for col in range(MAZE_COLS):
                if self.base_grid[row][col] == 0:
                    # expand cells
                    for i in range(CELL_EXPANSION):
                        for j in range(CELL_EXPANSION):
                            self.expanded_grid[row * CELL_EXPANSION + i][col * CELL_EXPANSION + j] = 0

    def get_random_white_cell(self):
        # find random white cell
        while True:
            row = random.randint(0, MAZE_ROWS-1)
            col = random.randint(0, MAZE_COLS-1)
            if self.expanded_grid[int((row+0.5)*CELL_EXPANSION)][int((col+0.5)*CELL_EXPANSION)] == 0:
                return (int((col+0.5)*CELL_EXPANSION), int((row+0.5)*CELL_EXPANSION))

    def draw(self):
        # draw the maze
        for row in range(MAZE_ROWS * CELL_EXPANSION):
            for col in range(MAZE_COLS * CELL_EXPANSION):
                color = WHITE if self.expanded_grid[row][col] == 0 else BLACK
                pygame.draw.rect(screen, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # draw the target
        target_x, target_y = self.target
        pygame.draw.rect(screen, GREEN, (target_x * CELL_SIZE, target_y * CELL_SIZE, CELL_SIZE, CELL_SIZE))


class Wheelchair:
    '''Wheelchair class. Controls the wheelchair and game dynamics.'''
    
    def __init__(self, maze):
        self.x, self.y = 1.5*CELL_SIZE*CELL_EXPANSION, 1.5*CELL_SIZE*CELL_EXPANSION
        self.direction = 0  # angle in degrees (0 is right)
        self.speed = 0
        self.maze = maze  
        self.score = INITIAL_SCORE  # initial score
        self.win = False  # win flag

    def can_move(self, dx, dy):
        # compute new position
        new_x = (self.x + dx) // CELL_SIZE
        new_y = (self.y + dy) // CELL_SIZE

        # check if new position is white and whithin the board
        if 0 <= new_x < len(self.maze.expanded_grid[0]) and 0 <= new_y < len(self.maze.expanded_grid):
            return self.maze.expanded_grid[int(new_y)][int(new_x)] == 0
        return False

    def move(self):
        if self.win:
            return  # after win, disallow movement

        # compute dx and dy based on the direction (angle in degrees)
        rad = math.radians(self.direction)
        dx = self.speed * math.cos(rad)
        dy = -self.speed * math.sin(rad)

        # check if can move before changing position
        if self.can_move(dx, dy):
            self.x += dx
            self.y += dy
        else:
            # count collision
            self.score += 1
            self.speed = 0

    def rotate(self, direction):
        if self.win:
            return  # stop rotation after winning
        if direction == "LEFT":
            self.direction = (self.direction + 45) % 360
        elif direction == "RIGHT":
            self.direction = (self.direction - 45) % 360

    def stop(self):
        self.speed = 0

    def execute_command(self, command):
        if command == "FORWARD":
            self.speed = CELL_SIZE // 2
        elif command == "BACKWARD":
            self.speed = -CELL_SIZE // 2
        elif command == "LEFT":
            self.rotate("LEFT")
        elif command == "RIGHT":
            self.rotate("RIGHT")
        elif command == "STOP":
            self.stop()

    def check_win(self):
        # check if target is reached
        target_x, target_y = self.maze.target
        return (self.x // CELL_SIZE, self.y // CELL_SIZE) == (target_x, target_y)

    def draw(self):
        # draw wheelchair as a triangle
        rad = math.radians(self.direction)
        front_x = self.x + CELL_SIZE * math.cos(rad)
        front_y = self.y - CELL_SIZE * math.sin(rad)
        left_x = self.x + CELL_SIZE * math.cos(rad + math.radians(135))
        left_y = self.y - CELL_SIZE * math.sin(rad + math.radians(135))
        right_x = self.x + CELL_SIZE * math.cos(rad - math.radians(135))
        right_y = self.y - CELL_SIZE * math.sin(rad - math.radians(135))

        pygame.draw.polygon(screen, BLUE, [(front_x, front_y), (left_x, left_y), (right_x, right_y)]) 


def sound_capture_thread(wheelchair):
    '''thread for capturing the microphone sound and converting it to a command to the wheelchair.'''
    
    RATE = 48000  # sampling rate of 16kHz
    RECORD_SECONDS = 1  # capture 1 second of sound
    OVERLAP_SECONDS = .95  # overlap between segments
    NO_OVERLAP_SECONDS = RECORD_SECONDS - OVERLAP_SECONDS # new data on each segment

    AMP_THRESHOLD = 0.2  # minimum amplitude to consider sound
    GAP_TIME_THRESHOLD = 0.5  # minimum time between classifications

    buffer_size = int(RATE * RECORD_SECONDS)
    buffer = np.zeros(buffer_size, dtype=np.float32)

    def callback(indata, frames, timestamp, status):
        global last_classification_time
        nonlocal buffer
        if status:
            print(status)
        
        # add the new data to the buffer, rolling the current data to the left
        buffer = np.roll(buffer, -len(indata))
        buffer[-len(indata):] = indata[:, 0]

        center_buffer = buffer[len(buffer)//2 - len(buffer)//10 : len(buffer)//2 + len(buffer)//10]
        if np.max(np.abs(center_buffer)) < AMP_THRESHOLD or (time.time() - last_classification_time) < GAP_TIME_THRESHOLD:
            return
        
        last_classification_time = time.time()
        
        # convert the captured sound to a command
        command = process_sound(buffer)

        if command:
            wheelchair.execute_command(command)

    # open audio stream from microphone
    with sd.InputStream(device=None, callback=callback, channels=1, samplerate=RATE, blocksize=int(RATE * NO_OVERLAP_SECONDS)):
        while True:
            time.sleep(NO_OVERLAP_SECONDS)  # Aguardar NO_OVERLAP_SECONDS


def main():
    '''Main function of the game.'''
    global start_recording_flag, start_recording_time
    
    clock = pygame.time.Clock()
    maze = Maze()
    wheelchair = Wheelchair(maze)
    running = True
    start_ticks = pygame.time.get_ticks()  # Tempo inicial

    # start sound capture thread
    sound_thread = threading.Thread(target=sound_capture_thread, args=(wheelchair,))
    sound_thread.daemon = True
    sound_thread.start()

    while running:
        screen.fill(WHITE)
        maze.draw()
        wheelchair.draw()

        # render points on screen
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Colisions: {wheelchair.score}", True, RED)
        screen.blit(score_text, (10, 10))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if wheelchair.win:
                    continue  # stop wheelchair control after winning
                if event.key == pygame.K_w:  # Forward
                    wheelchair.execute_command("FORWARD")
                elif event.key == pygame.K_s:  # Backward
                    wheelchair.execute_command("BACKWARD")
                elif event.key == pygame.K_a:  # Rotate left (45 degrees)
                    wheelchair.execute_command("LEFT")
                elif event.key == pygame.K_d:  # Rotate right (45 degrees)
                    wheelchair.execute_command("RIGHT")
                elif event.key == pygame.K_SPACE:  # Stop
                    wheelchair.execute_command("STOP")
                


        # check victory
        if wheelchair.check_win():
            wheelchair.win = True  # register victory
            font = pygame.font.SysFont(None, 44)
            win_text = font.render("Congratulations! You reached the target location!", True, GREEN)
            screen.blit(win_text, (SCREEN_WIDTH // 15, SCREEN_HEIGHT // 2))
            wheelchair.stop()  # stop the wheelchair

        wheelchair.move()

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


###############################################################################################
#                                                                                             #
#                              STUDENT CODE STARTS HERE                                       #
#                                                                                             #
# Edit, replace, change the following code as you wish. The only thing that                   #
# must be kept is the name of the function process_sound, and it must return                  #
# a string with a command for the game to act. The list of possible commands is:              #
# ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP", ""] -> empty string means do nothing       #
#                                                                                             #
###############################################################################################

commands = {
    "_unknown_": "", 
    "forward": "FORWARD", 
    "backward": "BACKWARD", 
    "left": "LEFT", 
    "right": "RIGHT", 
    "stop": "STOP"
}

from joblib import load

model = load("final_model.joblib")

scaler = load("scaler.joblib")


def classify(features):
    ''' receives a feature vector and returns a prediction from the classifier '''
    prediction = model.predict( features )
    return commands[ prediction[0] ]



import numpy as np
import librosa

def compute_dft(audio_data, sample_rate):
    n = len(audio_data)
    freqs = np.fft.rfftfreq(n, d=1/sample_rate)
    magnitude = np.abs(np.fft.rfft(audio_data))
    return freqs, magnitude

def extract_audio_features(segment, sample_rate):
    features = []
    feature_names = []

    mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=12)
    mfcc_mean = np.mean(mfccs, axis=1)
    features.extend(mfcc_mean)
    feature_names.extend([f"MFCC_{i+1}" for i in range(len(mfcc_mean))])

    zcr = np.mean(librosa.feature.zero_crossing_rate(segment))
    features.append(zcr)
    feature_names.append("Zero_Crossing_Rate")

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sample_rate))
    features.append(spectral_centroid)
    feature_names.append("Spectral_Centroid")

    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=segment))
    features.append(spectral_flatness)
    feature_names.append("Spectral_Flatness")

    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sample_rate))
    features.append(spectral_bandwidth)
    feature_names.append("Spectral_Bandwidth")

    stft = np.abs(librosa.stft(segment, n_fft=512))
    psd = np.mean(stft ** 2, axis=1)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=512)
    low_freq_idx = np.where(freqs < 500)[0]
    mid_freq_idx = np.where((freqs >= 500) & (freqs < 2000))[0]
    high_freq_idx = np.where(freqs >= 2000)[0]
    low_freq_energy = np.sum(psd[low_freq_idx])
    mid_freq_energy = np.sum(psd[mid_freq_idx])
    high_freq_energy = np.sum(psd[high_freq_idx])
    features.extend([low_freq_energy, mid_freq_energy, high_freq_energy])
    feature_names.extend(["Low_Freq_Energy", "Mid_Freq_Energy", "High_Freq_Energy"])

    return np.array(features, dtype=np.float32), feature_names

def extract_features(signal, sample_rate, segment_duration=0.2):
    segment_samples = int(segment_duration * sample_rate)
    total_segments = len(signal) // segment_samples
    segment_features_list = []
    feature_names = None

    for i in range(total_segments):
        segment = signal[i * segment_samples: (i + 1) * segment_samples]
        segment_features, segment_feature_names = extract_audio_features(segment, sample_rate)
        segment_features_list.append(segment_features)

        if feature_names is None:
            feature_names = segment_feature_names

    if segment_features_list:
        segment_features_array = np.array(segment_features_list)
        features_mean = np.mean(segment_features_array, axis=0)
        features_std = np.std(segment_features_array, axis=0)
        all_features = np.concatenate((features_mean, features_std))
        feature_names = [name + '_mean' for name in feature_names] + [name + '_std' for name in feature_names]
    else:
        # 如果没有片段，返回零数组
        all_features = np.zeros(len(feature_names) * 2)

    return all_features, feature_names


def extract_frequency_features(audio_data, sample_rate):

    freqs, magnitude = compute_dft(audio_data, sample_rate)

    features = []
    feature_names = []

    dominant_freq = freqs[np.argmax(magnitude)]
    features.append(dominant_freq)
    feature_names.append("dominant_freq")

    energy_0_500 = np.sum(magnitude[(freqs >= 0) & (freqs <= 500)])
    energy_500_1000 = np.sum(magnitude[(freqs > 500) & (freqs <= 1000)])
    energy_ratio = energy_0_500 / (energy_500_1000 + 1e-6)
    features.append(energy_ratio)
    feature_names.append("energy_ratio")

    weighted_freqs = magnitude * freqs
    spectral_centroid = np.sum(weighted_freqs) / np.sum(magnitude)
    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude))
    features.append(spectral_bandwidth)
    feature_names.append("spectral_bandwidth")

    return np.array(features, dtype=np.float32), feature_names

def combine_features(existing_features, new_features, existing_feature_names, new_feature_names):

    combined_features = np.concatenate([existing_features, new_features])
    combined_feature_names = existing_feature_names + new_feature_names
    return combined_features, combined_feature_names

def process_sound(sound):
    command = ""
    sound=sound[::3]
    audio_features, feature_names = extract_features(sound, sample_rate=16000, segment_duration=0.2)
    dft_features, dft_feature_names = extract_frequency_features(sound, sample_rate = 16000)
    combined_features, combined_feature_names = combine_features(
                existing_features=audio_features,
                new_features=dft_features,
                existing_feature_names=feature_names,
                new_feature_names=dft_feature_names
            )
    combined_features = np.array(combined_features).reshape(1, -1)
    combined_features = scaler.transform(combined_features)

    print(combined_features)  # Debug
    
    # Uncomment the follwing lines:    

    command = classify( combined_features )
    
    print(command)

    
    return command


if __name__ == "__main__":
    main()