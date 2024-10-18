import gymnasium as gym
import numpy as np
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from minihex.HexSingleGame import HexEnv
import time
import pygame
import random
import math
from minihex.SelfplayWrapper import selfplay_wrapper, BaseRandomPolicy
from minihex.interactive.interactive import InteractiveGame

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_action_mask()


def show_menu_screen():
    running = True
    level_selected = None
    number = 7

    while running:
        # background_image = pygame.transform.scale(background_image, (SCREEN_WIDTH, SCREEN_HEIGHT))
        # screen.blit(background_image, (0, 0))
        scale_background()

        title = font.render("Wähle eine Spielfeldgröße", True, BLACK)
        screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 100))

        easy_button = pygame.Rect(SCREEN_WIDTH//2 - 100, 375, 200, 50)
        medium_button = pygame.Rect(SCREEN_WIDTH//2 - 100, 300, 200, 50)
        hard_button = pygame.Rect(SCREEN_WIDTH//2 - 100, 225, 200, 50)
        start_button = pygame.Rect(SCREEN_WIDTH//2 - 100, 500, 200, 50)
        border_radius=15

        pygame.draw.rect(screen, ORANGE, easy_button, border_radius=border_radius)
        pygame.draw.rect(screen, WHITE, medium_button, border_radius=border_radius)
        pygame.draw.rect(screen, ORANGE, hard_button, border_radius=border_radius)
        pygame.draw.rect(screen, GREEN, start_button, border_radius=border_radius)

        easy_text = small_font.render("-", True, BLACK)
        medium_text = small_font.render(str(number) +" x " + str(number), True, BLACK)
        hard_text = small_font.render("+", True, BLACK)
        start_text = small_font.render("START", True, BLACK)

        # screen.blit(easy_text, (easy_button.x + 97, easy_button.y + 10))
        # screen.blit(medium_text, (medium_button.x + 60, medium_button.y + 10))
        # screen.blit(hard_text, (hard_button.x + 97, hard_button.y + 10))
        # screen.blit(start_text, (start_button.x + 60, start_button.y + 10))
        screen.blit(easy_text, (easy_button.x + (easy_button.width - easy_text.get_width()) // 2,
                                easy_button.y + (easy_button.height - easy_text.get_height()) // 2))
        screen.blit(medium_text, (medium_button.x + (medium_button.width - medium_text.get_width()) // 2,
                                medium_button.y + (medium_button.height - medium_text.get_height()) // 2))
        screen.blit(hard_text, (hard_button.x + (hard_button.width - hard_text.get_width()) // 2,
                                hard_button.y + (hard_button.height - hard_text.get_height()) // 2))
        screen.blit(start_text, (start_button.x + (start_button.width - start_text.get_width()) // 2,
                                start_button.y + (start_button.height - start_text.get_height()) // 2))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                quit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if easy_button.collidepoint(event.pos):
                    if number > 5:
                        number -= 1

                    # running = False
                # elif medium_button.collidepoint(event.pos):
                #     level_selected = 'medium'
                    # running = False
                elif hard_button.collidepoint(event.pos):
                    if number < 11:
                        number += 1
                    # running = False
                elif start_button.collidepoint(event.pos):
                    running = False

    return number

def show_start_screen():
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption('Start New Game')

    font = pygame.font.Font(None, 80)
    white = (255, 255, 255)
    black = (0, 0, 0)
    yellow = (255, 255, 0)

    clock = pygame.time.Clock()

    # Main loop for the start screen
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False  # Exit the game if the player closes the window
            if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                return True  # Start the game if the player presses any key or clicks

        screen.fill(white)
        text_surface = font.render("Press any key to start", True, black)
        text_rect = text_surface.get_rect(center=(400, 300))
        screen.blit(text_surface, text_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return False  # Return False to exit game if needed

def show_animation(winner):
    # Pygame logic to show the cheering animation
    # screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    # background_image = pygame.image.load("assets/hex-background.png").convert_alpha()

    # Get the original image size
    bg_width, bg_height = background_image.get_size()

    # Center the image if it's the same size as the screen or if no scaling is needed
    x_offset = (SCREEN_WIDTH - bg_width) // 2
    y_offset = (SCREEN_HEIGHT - bg_height) // 2
    #pygame.display.set_caption('Cheering Animation')

    font = pygame.font.Font(None, 80)
    white = (255, 255, 255)
    grey = (180, 180, 180)
    black = (0, 0, 0)
    yellow = (255, 255, 0)
    red = (251, 41, 67)
    blue = (6, 154, 243)

    def render_text_with_animation(text, base_font_size, frame_count):
        font_size = base_font_size + int(10 * math.sin(frame_count / 7))
        font = pygame.font.Font(None, font_size)
        if winner == "Blau":
            color = blue
        else:
            color = red
        text_surface = font.render(text, True, color)
        return text_surface

    clock = pygame.time.Clock()
    base_font_size = 80
    frame_count = 0
    animation_duration = 200  # Duration in frames

    running = True
    while running and frame_count < animation_duration:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # screen.blit(background_image, (0,0))
        scale_background()
        #screen.fill(white)

        text_surface = render_text_with_animation(f"Spieler {winner} hat gewonnen!", base_font_size, frame_count)
        text_rect = text_surface.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
        screen.blit(text_surface, text_rect)

        pygame.display.flip()
        frame_count += 1
        clock.tick(60)

    #pygame.quit()

def scale_background():

    # Get the original size of the background image
    bg_width, bg_height = background_image.get_size()

    # Calculate aspect ratios
    window_aspect_ratio = SCREEN_WIDTH / SCREEN_HEIGHT
    image_aspect_ratio = bg_width / bg_height

    # Determine the new size while maintaining aspect ratio
    if window_aspect_ratio > image_aspect_ratio:
        # Window is wider than the image's aspect ratio
        new_height = SCREEN_HEIGHT
        new_width = int(new_height * image_aspect_ratio)
    else:
        # Window is taller than the image's aspect ratio
        new_width = SCREEN_WIDTH
        new_height = int(new_width / image_aspect_ratio)

    # Scale the image to the new dimensions
    scaled_image = pygame.transform.scale(background_image, (new_width, new_height))

    # Calculate the position to center the image on the screen
    x_offset = (SCREEN_WIDTH - new_width) // 2
    y_offset = (SCREEN_HEIGHT - new_height) // 2

    # Blit the scaled image, cropping the edges as necessary
    screen.blit(scaled_image, (x_offset, y_offset))
##########################################################################################
##########################################################################################

# Initialize pygame
pygame.init()

# Screen settings
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
SCREEN_WIDTH, SCREEN_HEIGHT = screen.get_size() 

pygame.display.set_caption("Game with Levels")

# Fonts and colors
font = pygame.font.SysFont('Arial', 40)
small_font = pygame.font.SysFont('Arial', 30)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255,160, 0)

background_image = pygame.image.load("assets/hex-background.png").convert_alpha()

# Get the original image size
bg_width, bg_height = background_image.get_size()

# Center the image if it's the same size as the screen or if no scaling is needed
x_offset = (SCREEN_WIDTH - bg_width) // 2
y_offset = (SCREEN_HEIGHT - bg_height) // 2



while True:
    # background_image = pygame.transform.scale(background_image, (SCREEN_WIDTH, SCREEN_HEIGHT))
    scale_background()
    board_size = show_menu_screen()

    # screen.blit(background_image, (0, 0))
    env = selfplay_wrapper(HexEnv)(play_gui=True, 
                                    board_size=board_size, 
                                    scores=np.zeros(20),
                                    #    prob_model=model,
                                    agent_player_num=0)

    env = ActionMasker(env, mask_fn)


    state, info = env.reset()
    terminated = False

    player = InteractiveGame(state)

    while not terminated:
        scale_background()
        board = state

        env.opponent_model.gui.update_board(board)
        action = player.choose_action(board, env.legal_actions())

        state, reward, terminated, truncated, info = env.step(action)
    show_animation(info["winner"])