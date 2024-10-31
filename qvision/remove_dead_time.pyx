# qvision/remove_dead_time.pyx
import numpy as np
cimport numpy as np
cimport cython  # Aggiungi questa riga

@cython.boundscheck(False)
@cython.wraparound(False)
def remove_dead_time_cython(np.ndarray[double, ndim=1] times, double dead_time):
    """
    Rimuove i rilevamenti che avvengono ad un intervallo di tempo dal precedente inferiore al tempo morto.

    Args:
        times (np.ndarray): Array di tempi di rilevamento (assunto già ordinato).
        dead_time (float): Tempo morto.

    Returns:
        np.ndarray: Array di tempi di rilevamento dopo aver rimosso il tempo morto.
    """
    if len(times) == 0:
        return times

    cdef int n = len(times)
    cdef np.ndarray[double, ndim=1] cleaned = np.empty(n, dtype=np.float64)
    cleaned[0] = times[0]
    cdef int count = 1
    cdef double last_time = times[0]

    cdef int i
    cdef double t
    for i in range(1, n):
        t = times[i]
        if t >= last_time + dead_time:
            cleaned[count] = t
            count += 1
            last_time = t

    return cleaned[:count]