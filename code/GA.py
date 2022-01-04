import numpy as np
from cv2 import cv2
import math
import sys

# np.set_printoptions(threshold=sys.maxsize)

#
# def write(msg):
#     with open('output.txt', 'w') as f:
#         sys.stdout = f
#         print(msg)
#         sys.stdout = sys.stdout
#         f.close()


def read_image(image_name):
    img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE).astype('uint8')

    return img, img.shape[0], img.shape[1]

def img_to_chromosome(img, rows, cols):
    return np.reshape(img, rows * cols)


def chromosome_to_img(img, rows, cols):
    return np.reshape(img, (rows, cols))


#genereaza populatia initiala
def initial_population(pop_size, chromo_size, gray_levels):
    population = np.zeros((pop_size, chromo_size))
    for i in range(0, pop_size):

        population[i, :] = np.sort(np.random.randint(0, gray_levels, chromo_size))

    return population


def compute_sobel_edge(gray):
    # eliminare noise
    # https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Gradient_Sobel_Laplacian_Derivatives_Edge_Detection.php
    img = cv2.GaussianBlur(gray, (3, 3), 0)

    # convolutia
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    return sobel_x

# calculeza fitness-ul fiecarui cromozom
def fitness(img):
    rows = img.shape[0]
    cols = img.shape[1]

    edges = compute_sobel_edge(img)

    result = 1

    for x in range(1, rows - 1):
        for y in range(1, cols - 1):
            sol1 = img[x + 1][y - 1] + 2 * img[x + 1][y] + img[x + 1][y + 1] - img[x - 1][y - 1] - 2 * img[x + 1][y] \
                   - img[x - 1][y + 1]
            sol2 = img[x - 1][y + 1] + 2 * img[x][y + 1] + img[x + 1][y + 1] - img[x - 1][y - 1] - 2 * img[x][y - 1] \
                   - img[x + 1][y - 1]

            sol = math.sqrt((sol1 ** 2) + (sol2 ** 2))

            result = result + sol

    edge = 1
    for r in range(rows):
        for c in range(cols):
            if edges[r][c] != 0:
                edge += 1

    if result == 0:
        result = 1
    return math.log(math.log(result) * edge)


# operatia de crossover in 2 puncte
def crossover(parent1, parent2):
    size = min(len(parent1), len(parent2))

    cross_point1 = np.random.randint(0, size)
    cross_point2 = np.random.randint(0, size - 1)

    if cross_point1 > cross_point2:
        cross_point1, cross_point2 = cross_point2, cross_point1

    else:
        cross_point2 += 1

    parent_aux = parent1.copy()
    parent1[cross_point1:cross_point2] = parent2[cross_point1:cross_point2]
    parent2[cross_point1:cross_point2] = parent_aux[cross_point1:cross_point2]

    return parent1, parent2


def enhance(unique_values, chromo_size, img_chromo, i):

    # se face o copie a imaginii initiale si se lucreaza cu aceasta copie
    img_chromo1 = img_chromo.copy()

    # print("enumerate pop  ", id)
    # se face schimbul de date dintre cromozomul initial(imaginea initiala) si individul generat random
    for idx, unique in enumerate(unique_values):
        for j in range(chromo_size):

            if img_chromo1[j] == unique:
                img_chromo1[j] = i[idx]

    return img_chromo1


def perform_crossover(parentss, fitness_values, rows, cols, n, ps, pc):
    new_generation = []
    final_new_gen = []
    new_fitness = []
    for i in range(n):
        parent1_idx = np.random.randint(0, len(parentss))
        parent2_idx = np.random.randint(0, len(parentss))
        if parent1_idx == parent2_idx:
            parent2_idx = np.random.randint(0, len(parentss))

        # sunt in forma de cromozomi !!
        child1, child2 = crossover(parentss[parent1_idx], parentss[parent2_idx])
        new_generation.append(child1)
        new_generation.append(child2)

        # transforma cromozomii in imagini si calculeaza fitness si salveaza valorile
        img = chromosome_to_img(child1, rows, cols)
        fitness_chromo = fitness(img)
        new_fitness.append(fitness_chromo)

        img1 = chromosome_to_img(child2, rows, cols)
        fitness_chromo1 = fitness(img1)

        new_fitness.append(fitness_chromo1)

    for k in range(round(ps*pc)):
        max_f = max(new_fitness)
        for id, i in enumerate(new_fitness):
            if i == max_f:
                fitness_values.append(i)
                final_new_gen.append(new_generation[id])
                new_fitness.remove(i)
    print(final_new_gen.__len__())
    return final_new_gen


def show_image(image,rows, cols):
    new_img = chromosome_to_img(image, rows, cols)
    cv2.imshow("final", new_img)
    cv2.waitKey(0)
    cv2.imwrite("Final_img.jpg", new_img)
# ps = 50
# # crossover rate
# pc = 0.2
# num_crossovers = 8
#
# # citeste imaginea
# img = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE).astype('uint8')
# rows = img.shape[0]
# cols = img.shape[1]
#
# # dimensiunea cromozomului
# chromo_size = (img.shape[0] * img.shape[1])
#
# # dimensiunea populatiei initiale
# pop_size = ps
#
# # cate niveluri de gri sunt in imagine
# gray_levels = max(np.unique(img))
#
# # valori unice in imagine
# unique_values = np.unique(img)
#
# # genereaza pop initiala
# pop = initial_population(pop_size, len(unique_values), gray_levels)
#
# fitness_values = []
# enhanced_arr = []
#
# # transforma imaginea intr-un cromozom
# img_chromo = img_to_chromosome(img, rows, cols)
#
#
# for id, i in enumerate(pop):
#
#     #se face o copie a imaginii initiale si se lucreaza cu aceasta copie
#     img_chromo1 = img_chromo.copy()
#
#     print("enumerate pop  ", id)
#     # se face schimbul de date dintre cromozomul initial(imaginea initiala) si individul generat random
#     for idx, unique in enumerate(unique_values):
#         for j in range(chromo_size):
#
#             if img_chromo1[j] == unique:
#                 img_chromo1[j] = i[idx]
#
#     enhanced_img = chromosome_to_img(img_chromo1, rows, cols)
#
#     # se salveaza in forma de cromozom
#     enhanced_arr.append(img_chromo1)
#
#     # calculeaza fitness-ul
#     fit = fitness(enhanced_img)
#     fitness_values.append(fit)
#
#
# prev_greatest_fit = 0
# while True:
#
#     new_generation = perform_crossover(enhanced_arr,fitness_values, rows, cols, num_crossovers, ps, pc)
#     print("Len new gen: ", len(new_generation))
#
#     enhanced_arr.extend(new_generation)
#     greatest_fit = max(fitness_values)
#     idx = fitness_values.index(greatest_fit)
#     ps = ps - len(new_generation)
#
#     eps = 0.02 * greatest_fit
#
#     if greatest_fit - prev_greatest_fit < eps or ps == 1:
#         show_image(enhanced_arr[idx])
#         break
#
#     prev_greatest_fit = greatest_fit
