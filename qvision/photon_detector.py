import cupy as cp

# Controlla se una GPU è disponibile
if not cp.cuda.runtime.getDeviceCount():
    raise RuntimeError("No GPU found! Please ensure a compatible GPU is available.")

def calculate_transmissibility(lambda_ob):
    """
    Calculates Tob and Tpr transmissibility based on the image and the prob.

    Args:
        lambda_ob: CuPy array of pixel values.

    Returns:
        float, transmissibility of the SLM.
    """
    n_p = lambda_ob.size  # Numero di pixel
    cost = cp.max(cp.abs(lambda_ob))  # Usa CuPy per le operazioni
    return cp.sum(lambda_ob) / (n_p * cost)

def coinc(f, Rate, eta, tau, dcr, Delta_T, N_p=100, Rifl=0.5):
    """
    Simula il numero di coincidenze di fotoni rilevati.

    Args:
        f: float, modulo quadro del prodotto scalare delle funzioni d'onda associate ai due fotoni della coppia.
        Rate: float, numero di coppie di fotoni incidenti sul beamsplitter nell'unità di tempo (Hz).
        eta: CuPy array, efficienza dei detector (eta[0], eta[1]).
        tau: CuPy array, durata del tempo morto per i due detector (tau[0], tau[1]).
        dcr: CuPy array, dark count rate (Hz) per i due detector.
        Delta_T: float, durata di osservazione del singolo esperimento (sec).
        N_p: int, numero di ripetizioni dell'esperimento.
        Rifl: float, coefficiente di riflessione del beamsplitter.

    Returns:
        N_m: float, numero medio di coincidenze rilevate.
        N: CuPy array, distribuzione del numero di coppie rilevate.
        delt_t: float, intervallo di tempo medio fra le coincidenze.
        t_tol: float, tempo medio necessario per avere P*T coppie di fotoni incidenti sul beamsplitter.
    """
    # Impostazione dei parametri di default
    if dcr is None:
        dcr = cp.array([300, 300])
        Delta_T = 1
    elif Delta_T is None:
        Delta_T = 1

    Trasm = 1 - Rifl  # Coefficiente di trasmissione del beamsplitter
    N_phot = cp.round(Rate * 1.2 * Delta_T).astype(cp.int32)  # Numero di coppie di fotoni coinvolte
    N_dark = cp.round(cp.max(dcr) * 1.2 * Delta_T).astype(cp.int32)  # Numero di dark counts osservati
    N = cp.zeros(N_p)
    del_t = cp.zeros(N_p)
    t_tot = cp.zeros(N_p)

    # Coefficienti per la distribuzione di Poisson
    lambda_ = 1 / 100
    dt_p = lambda_ / Rate  # Campionamento del tempo (sec.)
    fact = 1 / cp.log(1 - lambda_)

    # Calcolo della probabilità di uscita
    lambda_dark = dcr * dt_p
    fact1 = 1 / cp.log(1 - lambda_dark[0])
    fact2 = 1 / cp.log(1 - lambda_dark[1])

    Pab = (1 - f) * (Trasm ** 2 + Rifl ** 2) + (Trasm - Rifl) ** 2 * f
    Paa = (1 + f) * Rifl * Trasm

    # Ciclo per la ripetizione dell'esperimento
    for l in range(N_p):
        # Generazione di sequenze casuali
        prob_posto = cp.random.rand(N_phot)
        prob1 = cp.random.rand(N_phot)
        prob2 = cp.random.rand(N_phot)

        # Inizializzazione dei rilevamenti per i detector
        dete1 = cp.zeros(N_phot)
        dete2 = cp.zeros(N_phot)
        dete1[((prob_posto < Pab) & (prob1 < eta[0])) | (
                ((prob_posto >= Pab) & (prob_posto < Pab + Paa)) & ((prob1 < eta[0]) | (prob2 < eta[0])))] = 1
        dete2[((prob_posto < Pab) & (prob2 < eta[1])) | (
            ((prob_posto >= Pab + Paa) & ((prob1 < eta[1]) | (prob2 < eta[1]))))] = 1

        # Tempo di arrivo delle coppie di fotoni
        situ = cp.random.rand(N_phot)
        nn = cp.ceil(fact * cp.log(1 - situ)).astype(cp.int32)
        dt_r = nn * dt_p
        t_a = cp.cumsum(dt_r)

        del_t[l] = cp.mean(cp.diff(t_a))  # Tempo medio fra le coppie
        t_tot[l] = t_a[-1]  # Tempo totale di arrivo

        # Tempi di rilevamento
        t_a1 = t_a[dete1 == 1]
        t_a2 = t_a[dete2 == 1]

        # Dark counts
        dark_prob1 = cp.random.rand(N_dark)
        dark_prob2 = cp.random.rand(N_dark)
        nd1 = cp.ceil(fact1 * cp.log(1 - dark_prob1)).astype(cp.int32)
        nd2 = cp.ceil(fact2 * cp.log(1 - dark_prob2)).astype(cp.int32)

        dt_d1 = nd1 * dt_p
        dt_d2 = nd2 * dt_p

        t_d1 = cp.cumsum(dt_d1)
        t_d2 = cp.cumsum(dt_d2)

        # Rilevamenti finali
        t_1 = cp.concatenate([t_a1, t_d1])
        t_1 = cp.sort(t_1[t_1 <= Delta_T])
        t_2 = cp.concatenate([t_a2, t_d2])
        t_2 = cp.sort(t_2[t_2 <= Delta_T])

        t_1 = remove_dead_time(t_1, tau[0])
        t_2 = remove_dead_time(t_2, tau[1])

        # Controllo delle coincidenze
        tt = cp.concatenate([t_1, t_2])
        tt = cp.sort(tt)

        # Controllo delle coincidenze solo se `tt` contiene almeno due elementi
        if tt.size > 1:
            N[l] = cp.sum(tt[:-1] == cp.roll(tt, -1)[:-1])

    N_m = cp.mean(N)
    return N_m, N, cp.mean(del_t), cp.mean(t_tot)

def remove_dead_time(times, dead_time):
    """
    Rimuove i rilevamenti che avvengono ad un intervallo di tempo dal precedente inferiore al tempo morto.
    """
    cleaned_times = []
    for i in range(len(times)):
        if i == 0 or times[i] - cleaned_times[-1] > dead_time:
            cleaned_times.append(times[i])
    return cp.array(cleaned_times)

def calculate_f_i(weights, Img, num_shots, ideal_conditions, non_ideal_parameters, f, N):
    """
    Calculates f_i based on non-ideal conditions.

    Args:
        weights: CuPy array, neuron weights.
        Img: CuPy array, source image.
        num_shots: int, number of shots.
        ideal_conditions: bool, whether conditions are ideal.
        non_ideal_parameters: dict, non-ideal parameters with keys 'eta', 'tau', 'drc' and 'C'.
        f: float, parameter used in the calculation.
        N: int, total number of pairs.

    Returns:
        f_i: float, computed value of f_i.
        N: int, the number of coincidences.
    """

    # Condizioni non ideali
    global N_m, N_p, P_i_ab, Rate, delta_T, Tob, Tpr
    if not ideal_conditions:
        eta = non_ideal_parameters.get('eta', cp.zeros(2))
        tau = non_ideal_parameters.get('tau', cp.zeros(2))
        drc = non_ideal_parameters.get('drc', 0.0)
        P = non_ideal_parameters.get('P', 0.0)
        C = non_ideal_parameters.get('C', 0.0)

        if N == 0:
            N = 1312.88
            f_i = 0
        else:
            Tob = calculate_transmissibility(Img)
            Tpr = calculate_transmissibility(weights)
            delta_T = C / (Tob * Tpr)
            Rate = P * Tob * Tpr
            N_m, _, _, _ = coinc(f, Rate, eta, tau, drc, delta_T, N_p=1, Rifl=0.5)
            P_i_ab = N_m / N
            f_i = 1 - 2 * P_i_ab
    else:
        f_i = f

    return f_i, N