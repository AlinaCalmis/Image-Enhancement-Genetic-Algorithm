from GA import *


def generate():
    # dimensiunea populatiei initiale
    ps = 50
    pop_size = ps

    # crossover rate
    pc = 0.2

    #numarul de crossover-uri
    num_crossovers = 8

    # citeste imaginea
    img, rows, cols = read_image('img_1.png')

    # dimensiunea cromozomului
    chromo_size = rows * cols

    # cate niveluri de gri sunt in imagine
    gray_levels = max(np.unique(img))

    # valori unice in imagine
    unique_values = np.unique(img)

    # genereaza pop initiala
    pop = initial_population(pop_size, len(unique_values), gray_levels)

    fitness_values = []
    enhanced_arr = []

    # transforma imaginea intr-un cromozom
    img_chromo = img_to_chromosome(img, rows, cols)

    for id, i in enumerate(pop):

        # imbunatatirea contrastului
        enhanced = enhance(unique_values, chromo_size, img_chromo, i)
        enhanced_img = chromosome_to_img(enhanced, rows, cols)

        # se salveaza in forma de cromozom
        enhanced_arr.append(enhanced)

        # calculeaza fitness-ul
        fit = fitness(enhanced_img)
        fitness_values.append(fit)

    prev_greatest_fit = 0
    while True:

        # creeaza noua generatie
        new_generation = perform_crossover(enhanced_arr, fitness_values, rows, cols, num_crossovers, ps, pc)

        # extinde vectorul de imagini imbunatite
        enhanced_arr.extend(new_generation)

        # cea mai mare valoare de fitness
        greatest_fit = max(fitness_values)
        idx = fitness_values.index(greatest_fit)

        # ps scade impreuna cu umarul de indivizi genarati in ultima generatie
        ps = ps - len(new_generation)

        # eps reprezinta diferenta minima intre fitnesul maxim generat intre 2 geberatii de cormozomi consecutive
        eps = 0.02 * greatest_fit

        # conditie de terminare
        if greatest_fit - prev_greatest_fit < eps or ps == 1:
            show_image(enhanced_arr[idx], rows, cols)
            break

        prev_greatest_fit = greatest_fit


if __name__ == '__main__':
    generate()
