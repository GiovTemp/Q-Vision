import numpy as np
from .utils import print_parameters


def calculate_transmissibility(lambda_ob):
    """
    Calculates Tob and Tpr transmissibility based on the image and the prob.
    Args:
        :param lambda_ob:

    Returns:
        np.sum(lamba) / (n_p * cost): float, transmissibility of the SLM .

    """

    n_p = lambda_ob.size  # Numero di pixel
    cost = np.max([np.max(lambda_ob), 1])

    return np.sum(lambda_ob) / (n_p * cost)


def calculate_photon_flow_pairs(Img, delta_T):
    max_intensity = np.max(Img)  # Trova il valore massimo nell'immagine
    return (np.sum(
        Img) / max_intensity * delta_T)  # Flusso normalizzato , diviso per delta T per rispettare il formato hz


def coinc(f, Rate, eta, tau, dcr, Delta_T, N_p=100, Rifl=0.5):
    """
    Simula il numero di coincidenze di fotoni rilevati.

    Args:
        f: float, modulo quadro del prodotto scalare delle funzioni d'onda associate ai due fotoni della coppia.
        Rate: float, numero di coppie di fotoni incidenti sul beamsplitter nell'unità di tempo (Hz).
        eta: list, efficienza dei detector (eta[0], eta[1]).
        tau: list, durata del tempo morto per i due detector (tau[0], tau[1]).
        dcr: list, dark count rate (Hz) per i due detector.
        Delta_T: float, durata di osservazione del singolo esperimento (sec).
        N_p: int, numero di ripetizioni dell'esperimento.
        Rifl: float, coefficiente di riflessione del beamsplitter.

    Returns:
        N_m: float, numero medio di coincidenze rilevate.
        N: numpy array, distribuzione del numero di coppie rilevate.
        delt_t: float, intervallo di tempo medio fra le coincidenze.
        t_tol: float, tempo medio necessario per avere P*T coppie di fotoni incidenti sul beamsplitter.
    """
    # Impostazione dei parametri di default
    if dcr is None:
        dcr = [300, 300]
        Delta_T = 1
    elif Delta_T is None:
        Delta_T = 1

    Trasm = 1 - Rifl  # Coefficiente di trasmissione del beamsplitter
    N_phot = round(Rate * 1.2 * Delta_T)  # Numero di coppie di fotoni coinvolte
    N_dark = round(max(dcr) * 1.2 * Delta_T)  # Numero di dark counts osservati
    N = np.zeros(N_p)
    del_t = np.zeros(N_p)
    t_tot = np.zeros(N_p)

    # Coefficienti per la distribuzione di Poisson
    lambda_ = 1 / 100
    dt_p = lambda_ / Rate  # Campionamento del tempo (sec.)
    fact = 1 / np.log(1 - lambda_)

    # Calcolo della probabilità di uscita
    lambda_dark = np.array(dcr) * dt_p
    fact1 = 1 / np.log(1 - lambda_dark[0])
    fact2 = 1 / np.log(1 - lambda_dark[1])

    Pab = (1 - f) * (Trasm ** 2 + Rifl ** 2) + (Trasm - Rifl) ** 2 * f
    Paa = (1 + f) * Rifl * Trasm

    # Ciclo per la ripetizione dell'esperimento
    for l in range(N_p):
        # Generazione di sequenze casuali
        prob_posto = np.random.rand(N_phot)
        prob1 = np.random.rand(N_phot)
        prob2 = np.random.rand(N_phot)

        # Inizializzazione dei rilevamenti per i detector
        dete1 = np.zeros(N_phot)
        dete2 = np.zeros(N_phot)
        dete1[((prob_posto < Pab) & (prob1 < eta[0])) | (
                ((prob_posto >= Pab) & (prob_posto < Pab + Paa)) & ((prob1 < eta[0]) | (prob2 < eta[0])))] = 1
        dete2[((prob_posto < Pab) & (prob2 < eta[1])) | (
            ((prob_posto >= Pab + Paa) & ((prob1 < eta[1]) | (prob2 < eta[1]))))] = 1

        # Tempo di arrivo delle coppie di fotoni
        situ = np.random.rand(N_phot)
        nn = np.ceil(fact * np.log(1 - situ)).astype(int)
        dt_r = nn * dt_p
        t_a = np.cumsum(dt_r)

        del_t[l] = np.mean(np.diff(t_a))  # Tempo medio fra le coppie
        t_tot[l] = t_a[-1]  # Tempo totale di arrivo

        # Tempi di rilevamento
        t_a1 = t_a[dete1 == 1]
        t_a2 = t_a[dete2 == 1]

        # Dark counts
        dark_prob1 = np.random.rand(N_dark)
        dark_prob2 = np.random.rand(N_dark)
        nd1 = np.ceil(fact1 * np.log(1 - dark_prob1)).astype(int)
        nd2 = np.ceil(fact2 * np.log(1 - dark_prob2)).astype(int)

        dt_d1 = nd1 * dt_p
        dt_d2 = nd2 * dt_p

        t_d1 = np.cumsum(dt_d1)
        t_d2 = np.cumsum(dt_d2)

        # Rilevamenti finali
        t_1 = np.concatenate([t_a1, t_d1])
        t_1 = np.sort(t_1[t_1 <= Delta_T])
        t_2 = np.concatenate([t_a2, t_d2])
        t_2 = np.sort(t_2[t_2 <= Delta_T])

        # Rimozione dei rilevamenti basati su tempo morto
        t_1 = remove_dead_time(t_1, tau[0])
        t_2 = remove_dead_time(t_2, tau[1])

        # Controllo delle coincidenze
        tt = np.concatenate([t_1, t_2])
        tt = np.sort(tt)
        N[l] = np.sum(tt == np.roll(tt, -1))  # Numero di coppie rilevate

    N_m = np.mean(N)
    return N_m, N, np.mean(del_t), np.mean(t_tot)


def coinc2(f, Rate, eta, tau, dcr, Delta_T, N_p=100, Rifl=0.5):
    # Impostazione dei parametri di default
    if dcr is None:
        dcr = [300, 300]
        Delta_T = 1
    elif Delta_T is None:
        Delta_T = 1

    Trasm = 1 - Rifl  # Coefficiente di trasmissione del beamsplitter
    N_phot = max(0, round(Rate * 1.2 * Delta_T))
    #N_phot = round(Rate * 1.2 * Delta_T)  # Numero di coppie di fotoni coinvolte
    N_dark = round(max(dcr) * 1.2 * Delta_T)  # Numero di dark counts osservati
    N = np.zeros(N_p)
    del_t = np.zeros(N_p)
    t_tot = np.zeros(N_p)

    lambda_ = 1 / 100
    dt_p = lambda_ / Rate  # Campionamento del tempo (sec.)
    fact = 1 / np.log(1 - lambda_)

    # Calcolo della probabilità di uscita
    lambda_dark = np.array(dcr) * dt_p
    fact1 = 1 / np.log(1 - lambda_dark[0])
    fact2 = 1 / np.log(1 - lambda_dark[1])

    Pab = (1 - f) * (Trasm ** 2 + Rifl ** 2) + (Trasm - Rifl) ** 2 * f
    Paa = (1 + f) * Rifl * Trasm

    # Ciclo per la ripetizione dell'esperimento
    for l in range(N_p):
        # Generazione di sequenze casuali
        prob_posto = np.random.rand(N_phot)
        prob1 = np.random.rand(N_phot)
        prob2 = np.random.rand(N_phot)

        dete1 = np.zeros(N_phot)
        dete2 = np.zeros(N_phot)
        dete1[((prob_posto < Pab) & (prob1 < eta[0])) | (
                ((prob_posto >= Pab) & (prob_posto < Pab + Paa)) & ((prob1 < eta[0]) | (prob2 < eta[0])))] = 1
        dete2[((prob_posto < Pab) & (prob2 < eta[1])) | (
            ((prob_posto >= Pab + Paa) & ((prob1 < eta[1]) | (prob2 < eta[1]))))] = 1

        situ = np.random.rand(N_phot)
        nn = np.ceil(fact * np.log(1 - situ)).astype(int)
        dt_r = nn * dt_p
        t_a = np.cumsum(dt_r)

        del_t[l] = np.mean(np.diff(t_a))
        t_tot[l] = t_a[-1] if len(t_a) > 0 else 0

        t_a1 = t_a[dete1 == 1]
        t_a2 = t_a[dete2 == 1]

        dark_prob1 = np.random.rand(N_dark)
        dark_prob2 = np.random.rand(N_dark)
        nd1 = np.ceil(fact1 * np.log(1 - dark_prob1)).astype(int)
        nd2 = np.ceil(fact2 * np.log(1 - dark_prob2)).astype(int)

        dt_d1 = nd1 * dt_p
        dt_d2 = nd2 * dt_p

        t_d1 = np.cumsum(dt_d1)
        t_d2 = np.cumsum(dt_d2)

        t_1 = np.concatenate([t_a1, t_d1])
        t_1 = np.sort(t_1[t_1 <= Delta_T])
        t_2 = np.concatenate([t_a2, t_d2])
        t_2 = np.sort(t_2[t_2 <= Delta_T])

        t_1 = remove_dead_time(t_1, tau[0])
        t_2 = remove_dead_time(t_2, tau[1])

        tt = np.concatenate([t_1, t_2])
        tt = np.sort(tt)

        # Controllo delle coincidenze solo se `tt` contiene almeno due elementi
        if len(tt) > 1:
            N[l] = np.sum(tt[:-1] == np.roll(tt, -1)[:-1])

    N_m = np.mean(N)
    return N_m, N, np.mean(del_t), np.mean(t_tot)


def remove_dead_time(times, dead_time):
    """
    Rimuove i rilevamenti che avvengono ad un intervallo di tempo dal precedente inferiore al tempo morto.
    """
    cleaned_times = []
    for i in range(len(times)):
        if i == 0 or times[i] - cleaned_times[-1] > dead_time:
            cleaned_times.append(times[i])
    return np.array(cleaned_times)


def calculate_f_i(weights, Img, num_shots, ideal_conditions, non_ideal_parameters, f, N):
    """
    Calculates f_i based on non-ideal conditions.

    Args:
        weights: numpy array, neuron weights.
        Img: numpy array, source image.
        num_shots: int, number of shots.
        ideal_conditions: bool, whether conditions are ideal.
        non_ideal_parameters: dict, non-ideal parameters with keys 'eta', 'tau', 'drc' and 'C'.
        f: float, parameter used in the calculation.

    Returns:
        f_i: float, computed value of f_i.
        :param N:
    """

    # Condizioni non ideali
    global N_m, N_p, P_i_ab, Rate, delta_T, Tob, Tpr
    if not ideal_conditions:
        eta = non_ideal_parameters.get('eta', 0.0)
        tau = non_ideal_parameters.get('tau', 0.0)
        drc = non_ideal_parameters.get('drc', 0.0)  # Valore di drc, ma non utilizzato nell'algoritmo
        P = non_ideal_parameters.get('P', 0.0)
        C = non_ideal_parameters.get('C', 0.0)

        if N == 0:
            #Calcolo coinc2 quando f = 0 solo al primo run
            #Solo la prima volta
            Tob = 0.5
            Tpr = 0.5
            delta_T = C / Tob * Tpr  # Calcolo di delta_T
            Rate = P / 4
            N, _, _, _ = coinc2(0, Rate, eta, tau, drc, 1, N_p=100, Rifl=0.5)
            N_m = N
            P_i_ab = N_m / N  # Calcolo del numero medio di coppie di fotoni
            # Ottengo il numero di rivelazioni totali
        else:
            # Quando usiamo le immagini
            Tob = calculate_transmissibility(Img)  # Calcolo della trasmissibilità
            Tpr = calculate_transmissibility(weights)  # Calcolo della trasmissibilità del probe
            delta_T = C / Tob * Tpr  # Calcolo di delta_T
            Rate = P * Tob * Tpr  # Calcolo del rate
            if f<0:
                f=f*-1 #inverto f perche al primo run mi da -1
            N_m, _, _, _ = coinc2(f, Rate, eta, tau, drc, 1, N_p=1, Rifl=0.5)
            P_i_ab = N_m / N  # Calcolo del numero medio di coppie di fotoni
        f_i = 1 - 2 * P_i_ab  # Calcolo finale di f_i
    else:
        f_i = f

    # Stampa dei parametri in formato tabellare
    parameters = [
        ["Parameter", "Value"],
        ["Weights", weights.tolist()],
        ["Image Shape", Img.shape],
        ["Num Shots", num_shots],
        ["Ideal Conditions", ideal_conditions],
        ["C", non_ideal_parameters.get('C', 0.0)],
        ["Eta", non_ideal_parameters.get('eta', [0.0, 0.0])],
        ["Tau", non_ideal_parameters.get('tau', [0.0, 0.0])],
        ["P", non_ideal_parameters.get('P', 0.0)],
        ["Tob", Tob],
        ["Tpr", Tpr],
        ["N", N],
        ["N_m", N_m],
        ["Rate", Rate],
        ["delta_T", delta_T],
        ["P_i_ab", P_i_ab],
        ["f", f],
        ["f_i", f_i],
    ]
    #print_parameters(parameters)
    return f_i, N
