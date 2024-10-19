import numpy as np


def calculate_transmissibility(lamba):
    """
    Calculates Tob and Tpr transmissibility based on the image and the prob.
    Args:
        :param lamba:

    Returns:
        np.sum(lamba) / (n_p * cost): float, transmissibility of the SLM .

    """

    n_p = lamba.size  # Numero di pixel
    cost = np.max(lamba)  # Valore massimo dell'intensità del pixel

    return np.sum(lamba) / (n_p * cost)


def calculate_photon_flow_pairs(Img, delta_T):
    max_intensity = np.max(Img)  # Trova il valore massimo nell'immagine
    return (np.sum(
        Img) / max_intensity * delta_T)  # Flusso normalizzato , diviso per delta T per rispettare il formato hz


import numpy as np


def coinc(f, Rate, eta, tau, T=None, N_p=100, Rifl=0.5):
    """
    Funzione per simulare il numero di coppie di fotoni rilevate.

    Args:
        f: float, modulo quadro del prodotto scalare delle funzioni d'onda associate ai due fotoni della coppia.
        Rate: float, numero di coppie di fotoni emessi nell'unità di tempo (Hz).
        eta: list, vettore efficienza dei detector (rapporto fra il numero di fotoni rilevati sul numero di fotoni incidenti).
        tau: list, vettore durata del tempo morto per i due detector (secondi).
        T: float, durata di osservazione del singolo esperimento (secondi).
        N_p: int, numero di ripetizioni dell'esperimento.
        Rifl: float, coefficiente di riflessione del beamsplitter.

    Returns:
        N_m: float, numero medio di coppie rilevate.
        N: array, distribuzione del numero di coppie rilevate nelle N_p ripetizioni.
        detec: array, numero medio di rilevazioni di fotoni per ogni singolo detector.
        delt_t: float, intervallo di tempo medio fra l'incidenza sul beamsplitter di una coppia di fotoni con quella successiva (secondi).
        t_tol: float, tempo medio necessario per avere P*T coppie di fotoni incidenti sul beamsplitter (salvo fluttuazioni coincide con T) (secondi).
    """

    # Se non viene specificato T, impostalo a 1
    if T is None:
        T = 1

    # Coefficiente di trasmissione del beamsplitter
    Trasm = 1 - Rifl

    # Numero totale di coppie di fotoni coinvolte nell'esperimento
    N_phot = round(Rate * T)

    # Inizializzazione delle variabili
    N = np.zeros(N_p)  # Distribuzione del numero di coppie rilevate
    dete = np.zeros((2, N_p))  # Rilevazioni per i due detector
    del_t = np.zeros(N_p)  # Intervalli di tempo fra le coppie di fotoni
    t_tot = np.zeros(N_p)  # Tempo totale di arrivo delle coppie di fotoni

    # Coefficiente della distribuzione di Poisson nella frequenza dell'arrivo di coppie di fotoni
    lambda_ = 1 / 100

    # Campionamento del tempo (sec.)
    dt_p = lambda_ / Rate

    # Fattore di scala per la distribuzione di Poisson
    fact = 1 / np.log(1 - lambda_)

    # Probabilità di uscita da porte diverse del beamsplitter
    C1 = (1 - f) * (Trasm ** 2 + Rifl ** 2) + (Trasm - Rifl) ** 2 * f
    C2 = (1 + f) * Rifl * Trasm  # Probabilità di uscita dalla stessa porta

    # Ciclo per le ripetizioni dell'esperimento
    for l in range(N_p):
        # Generazione di N_phot numeri casuali, distribuzione uniforme fra 0 e 1
        prob_posto = np.random.rand(N_phot)

        # Genera numeri casuali per il rilevamento
        prob1 = np.random.rand(N_phot)  # Primo detector
        prob2 = np.random.rand(N_phot)  # Secondo detector

        # Inizializzazione dei rilevamenti per i detector
        dete1 = np.zeros(N_phot)
        dete2 = np.zeros(N_phot)

        # Determina i rilevamenti sul primo detector
        dete1[(prob_posto < C1) & (prob1 < eta[0])] = 1
        dete1[(prob_posto >= C1) & (prob_posto < C1 + C2) & ((prob1 < eta[0]) | (prob2 < eta[0]))] = 1

        # Determina i rilevamenti sul secondo detector
        dete2[(prob_posto < C1) & (prob2 < eta[1])] = 1
        dete2[(prob_posto >= C1 + C2) & ((prob1 < eta[1]) | (prob2 < eta[1]))] = 1

        # Genera sequenze di tempi casuali
        situ = np.random.rand(N_phot)
        nn = np.ceil(fact * np.log(1 - situ)).astype(int)  # Intervalli di tempo tra le coppie

        # Calcola il tempo di attesa fra le coppie di fotoni
        dt_r = nn * dt_p
        t_a = np.zeros(N_phot)
        for al in range(N_phot):
            t_a[al] = np.sum(dt_r[:al + 1])  # Tempo di arrivo della al-esima coppia

        del_t[l] = np.mean(t_a[1:] - t_a[:-1])  # Tempo medio fra le coppie
        t_tot[l] = t_a[-1]  # Tempo di arrivo dell'ultima coppia di fotoni

        # Rimuovi i rilevamenti in base al tempo morto per il primo detector
        I_viv1 = np.where(dete1 == 1)[0]
        j = 1
        while j < len(I_viv1):
            if t_a[I_viv1[j]] - t_a[I_viv1[j - 1]] <= tau[0]:
                I_viv1 = np.delete(I_viv1, j)
            else:
                j += 1
        dete1 = np.zeros_like(prob_posto)
        dete1[I_viv1] = 1

        # Rimuovi i rilevamenti in base al tempo morto per il secondo detector
        I_viv2 = np.where(dete2 == 1)[0]
        j = 1
        while j < len(I_viv2):
            if t_a[I_viv2[j]] - t_a[I_viv2[j - 1]] <= tau[1]:
                I_viv2 = np.delete(I_viv2, j)
            else:
                j += 1
        dete2 = np.zeros_like(prob_posto)
        dete2[I_viv2] = 1

        N[l] = np.sum(dete1 * dete2)  # Numero di coppie rilevate
        dete[:, l] = np.array([np.sum(dete1), np.sum(dete2)])  # Rilevamenti per entrambi i detector

    N_m = np.mean(N)  # Numero medio di coppie rilevate
    detec = np.mean(dete, axis=1)  # Numero medio di rilevazioni per ogni detector
    delt_t = np.mean(del_t)  # Tempo medio fra le coppie
    t_tol = np.mean(t_tot)  # Tempo totale

    return N_m, N, detec, delt_t, t_tol


def calculate_f_i(weights, Img, num_shots, ideal_conditions, non_ideal_parameters, f):
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
    """

    # Condizioni non ideali
    if not ideal_conditions:
        eta = non_ideal_parameters.get('eta', 0.0)
        tau = non_ideal_parameters.get('tau', 0.0)
        drc = non_ideal_parameters.get('drc', 0.0)  # Valore di drc, ma non utilizzato nell'algoritmo

        # Se f è zero, impostiamo Tpr e Tob a 0.5
        if f == 0:
            Tpr = 0.5  # Trasmissibilità
            Tob = 0.5
            delta_T = 4 * non_ideal_parameters.get('C', 0.0)
            P = non_ideal_parameters.get('P', 0.0)
            Rate = P / 4
            N_p = 1
        else:
            # Calcolo per condizioni non ideali
            norm = np.sqrt(np.sum(np.square(weights)))  # Normalizzazione dei pesi
            Probe = np.square(weights / norm)  # Calcolo del probe
            Tob = calculate_transmissibility(Img)  # Calcolo della trasmissibilità
            Tpr = calculate_transmissibility(Probe)  # Calcolo della trasmissibilità del probe
            delta_T = non_ideal_parameters.get('C', 0.0) / Tob * Tpr  # Calcolo di delta_T
            P = non_ideal_parameters.get('P', 0.0)  # Calcolo del flusso di coppia
            Rate = P * Tob * Tpr  # Calcolo del rate
            N_p = 1  # Numero di ripetizioni

        # Chiamata alla funzione coinc per ottenere N_m
        N_m, _, _, _, _ = coinc(f, Rate, eta, tau, delta_T, N_p=N_p, Rifl=0.5)

        P_i_ab = N_m / num_shots  # Calcolo del numero medio di coppie di fotoni
        f_i = 1 - 2 * P_i_ab  # Calcolo finale di f_i
    else:
        f_i = f  # Se le condizioni sono ideali, restituiamo f direttamente

    return f_i
