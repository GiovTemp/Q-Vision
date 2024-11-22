import numpy as np
from .utils import print_parameters

from .remove_dead_time import remove_dead_time_cython



def calculate_transmissibility(lambda_ob):
    """
    Calculates Tob and Tpr transmissibility based on the image and the prob.
    Args:
        :param lambda_ob:

    Returns:
        np.sum(lamba) / (n_p * cost): float, transmissibility of the SLM .

    """

    n_p = lambda_ob.size  # Numero di pixel
    #print(f"N_P: {n_p}")
    cost = np.max([np.max(np.abs(lambda_ob)), 1])
    #print(f"Cost: {cost}")
    return np.sum(lambda_ob) / (n_p * cost)

def coinc2_optimized(f, Rate, eta, tau, dcr, Delta_T, N_p=100, Rifl=0.5):
    """
    Simula il numero di coincidenze di fotoni rilevati in modo ottimizzato usando solo NumPy.

    Args:
        f: float, modulo quadro del prodotto scalare delle funzioni d'onda associate ai due fotoni della coppia.
        Rate: float, numero di coppie di fotoni incidenti sul beamsplitter nell'unità di tempo (Hz).
        eta: array-like, efficienza dei detector (eta[0], eta[1]).
        tau: array-like, durata del tempo morto per i due detector (tau[0], tau[1]).
        dcr: array-like, dark count rate (Hz) per i due detector.
        Delta_T: float, durata di osservazione del singolo esperimento (sec).
        N_p: int, numero di ripetizioni dell'esperimento.
        Rifl: float, coefficiente di riflessione del beamsplitter.

    Returns:
        N_m: float, numero medio di coincidenze rilevate.
        N: array, distribuzione del numero di coppie rilevate.
        delt_t: float, intervallo di tempo medio fra le coincidenze.
        t_tot: float, tempo medio necessario per avere P*T coppie di fotoni incidenti sul beamsplitter.
    """
    # Impostazione dei parametri di default
    if dcr is None:
        dcr = [300, 300]
    if Delta_T is None:
        Delta_T = 1

    Trasm = 1 - Rifl  # Coefficiente di trasmissione del beamsplitter
    N_phot = int(round(Rate * 1.2 * Delta_T))  # Numero di coppie di fotoni coinvolte
    N_dark = int(round(max(dcr) * 1.2 * Delta_T))  # Numero di dark counts osservati

    # Pre-allocazione degli array per i risultati
    N = np.zeros(N_p, dtype=np.int32)
    del_t = np.zeros(N_p, dtype=np.float32)
    t_tot = np.zeros(N_p, dtype=np.float32)

    # Coefficienti per la distribuzione di Poisson
    lambda_ = 1 / 100
    dt_p = lambda_ / Rate  # Campionamento del tempo (sec.)
    fact = 1 / np.log(1 - lambda_)

    # Calcolo della probabilità di uscita
    lambda_dark = np.array(dcr) * dt_p
    epsilon = 1e-10  # Evita log(0)
    lambda_dark = np.clip(lambda_dark, epsilon, 1 - epsilon)
    fact1 = 1 / np.log(1 - lambda_dark[0])
    fact2 = 1 / np.log(1 - lambda_dark[1])

    Pab = (1 - f) * (Trasm ** 2 + Rifl ** 2) + (Trasm - Rifl) ** 2 * f
    Paa = (1 + f) * Rifl * Trasm
    Pab_plus_Paa = Pab + Paa  # Precalcolo per ottimizzare

    # Precompute constants per efficienza e tempo morto
    eta0 = eta[0]
    eta1 = eta[1]
    tau0 = tau[0]
    tau1 = tau[1]

    # Loop principale per le ripetizioni dell'esperimento
    for l in range(N_p):
        # Generazione di sequenze casuali
        prob_posto = np.random.rand(N_phot)
        prob1 = np.random.rand(N_phot)
        prob2 = np.random.rand(N_phot)

        # Calcolo dete1 e dete2 utilizzando operazioni booleane vettoriali
        dete1 = ((prob_posto < Pab) & (prob1 < eta0)) | (
            ((prob_posto >= Pab) & (prob_posto < Pab_plus_Paa)) & ((prob1 < eta0) | (prob2 < eta0))
        )
        dete2 = ((prob_posto < Pab) & (prob2 < eta1)) | (
            ((prob_posto >= Pab_plus_Paa) & ((prob1 < eta1) | (prob2 < eta1)))
        )

        # Generazione dei tempi di arrivo
        situ = np.random.rand(N_phot)
        situ = np.clip(situ, epsilon, 1 - epsilon)  # Evita log(0)
        nn = np.ceil(fact * np.log(1 - situ)).astype(int)
        dt_r = nn * dt_p
        t_a = np.cumsum(dt_r)

        # Calcolo del tempo medio fra le coppie e del tempo totale di arrivo
        if len(t_a) > 1:
            del_t[l] = np.mean(np.diff(t_a))
        else:
            del_t[l] = 0.0
        t_tot[l] = t_a[-1] if len(t_a) > 0 else 0.0

        # Tempi di rilevamento per ciascun detector
        t_a1 = t_a[dete1]
        t_a2 = t_a[dete2]

        # Generazione dei dark counts
        if N_dark > 0:
            dark_prob1 = np.random.rand(N_dark)
            dark_prob2 = np.random.rand(N_dark)
            dark_prob1 = np.clip(dark_prob1, epsilon, 1 - epsilon)  # Evita log(0)
            dark_prob2 = np.clip(dark_prob2, epsilon, 1 - epsilon)
            nd1 = np.ceil(fact1 * np.log(1 - dark_prob1)).astype(int)
            nd2 = np.ceil(fact2 * np.log(1 - dark_prob2)).astype(int)

            dt_d1 = nd1 * dt_p
            dt_d2 = nd2 * dt_p

            t_d1 = np.cumsum(dt_d1)
            t_d2 = np.cumsum(dt_d2)
        else:
            t_d1 = np.array([])
            t_d2 = np.array([])

        # Rilevamenti finali per ciascun detector
        if len(t_a1) > 0 or len(t_d1) > 0:
            t_1 = np.concatenate([t_a1, t_d1])
            t_1 = t_1[t_1 <= Delta_T]
            t_1_sorted = np.sort(t_1)
        else:
            t_1_sorted = np.array([])

        if len(t_a2) > 0 or len(t_d2) > 0:
            t_2 = np.concatenate([t_a2, t_d2])
            t_2 = t_2[t_2 <= Delta_T]
            t_2_sorted = np.sort(t_2)
        else:
            t_2_sorted = np.array([])

        # Rimozione del tempo morto
        if len(t_1_sorted) > 0:
            t_1_clean = remove_dead_time_cython(t_1_sorted, tau0)
        else:
            t_1_clean = np.array([])

        if len(t_2_sorted) > 0:
            t_2_clean = remove_dead_time_cython(t_2_sorted, tau1)
        else:
            t_2_clean = np.array([])

        # Concatenazione e ordinamento dei tempi per la coincidenza
        if len(t_1_clean) > 0 or len(t_2_clean) > 0:
            tt = np.concatenate([t_1_clean, t_2_clean])
            tt_sorted = np.sort(tt)
        else:
            tt_sorted = np.array([])

        # Controllo delle coincidenze solo se `tt_sorted` contiene almeno due elementi
        if len(tt_sorted) > 1:
            # Conta delle coincidenze: differenza zero tra tempi consecutivi
            # Utilizza una maschera booleana per identificare le coincidenze
            coincidences = np.sum(tt_sorted[1:] - tt_sorted[:-1] == 0)
            N[l] = coincidences
        else:
            N[l] = 0

    # Calcolo dei risultati finali
    N_m = np.mean(N)
    return N_m, N, np.mean(del_t), np.mean(t_tot)



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
        N = non_ideal_parameters.get('N',1312.88)
        #print(f"eta: {eta}")
        #print(f"tau: {tau}")
        #print(f"drc: {drc}")
        #print(f"P: {P}")
        #print(f"C: {C}")

        if N == 0:
            #Calcolo coinc2 quando f = 0 solo al primo run
            #Solo la prima volta
            #Tob = 0.5
            #Tpr = 0.5
            #delta_T = C / (Tob * Tpr)  # Calcolo di delta_T
            #print(f"deltaT: {delta_T}")
            #Rate = P / 4
            #print(f"Rate: {Rate}")
            #N, _, _, _ = coinc2(0, Rate, eta, tau, drc, delta_T, N_p=100, Rifl=0.5)
            #print(f"N: {N}")

            f_i = 0  # Calcolo finale di f_i
            #print(f"f_i: {f_i}")
            # Ottengo il numero di rivelazioni totali
        else:
            # Quando usiamo le immagini
            Tob = calculate_transmissibility(Img)  # Calcolo della trasmissibilità
            #print(f"Tob: {Tob}")
            Tpr = calculate_transmissibility(weights)  # Calcolo della trasmissibilità del probe
            #print(f"Tpr: {Tpr}")
            delta_T = C / (Tob * Tpr)  # Calcolo di delta_T
            #print(f"deltaT: {delta_T}")
            Rate = P * Tob * Tpr  # Calcolo del rate
            #print(f"Rate: {Rate}")
            #print("in coinc")
            N_m, _, _, _ = coinc2_optimized(f, Rate, eta, tau, drc, delta_T, N_p=1, Rifl=0.5)
            #print("out coinc")
            #print(f"N_m: {N_m}")
            P_i_ab = N_m / N  # Calcolo del numero medio di coppie di fotoni
            #print(f"P_i_ab: {P_i_ab}")
            f_i = 1 - 2 * P_i_ab  # Calcolo finale di f_i
            #print(f"f_i: {f_i}")
    else:
        f_i = f

    return f_i, N
