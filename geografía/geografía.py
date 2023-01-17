import matplotlib.pyplot as plt
import numpy as np
import shapefile as sf
from matplotlib import colors, cm

from modelo import Modelo


class Geografía(object):
    def __init__(símismo, archivo: str, país: str):
        símismo.forma = sf.Reader(archivo)
        símismo.país = país

    def dibujar(símismo, modelo: Modelo, llenar=True, alpha=1):
        fig, eje = plt.plot(figsize=(12, 6))

        traza = modelo.obt_traza_por_categoría(símismo.país)
        vals_por_región = traza.mean()

        escala = (min(vals_por_región), max(vals_por_región))
        if escala[0] == escala[1]:
            escala = (escala[0] - 0.5, escala[0] + 0.5)

        vals_norm = (vals_por_región - escala[0]) / (escala[1] - escala[0])

        escala_colores = símismo._resolver_colores()
        d_clrs = _gen_d_mapacolores(colores=escala_colores)

        mapa_color = colors.LinearSegmentedColormap('mapa_color', d_clrs)
        norm = colors.Normalize(vmin=escala[0], vmax=escala[1])
        cpick = cm.ScalarMappable(norm=norm, cmap=mapa_color)
        cpick.set_array(np.array([]))

        v_cols = mapa_color(vals_norm)
        v_cols[np.isnan(vals_norm)] = 1

        for i, frm in enumerate(símismo.forma.shapes()):
            puntos = frm.points
            partes = frm.parts

            for ip, i0 in enumerate(partes):  # Para cada parte de la imagen

                if ip < len(partes) - 1:
                    i1 = partes[ip + 1] - 1
                else:
                    i1 = len(puntos)

                seg = puntos[i0:i1 + 1]
                x_lon = np.zeros((len(seg), 1))
                y_lat = np.zeros((len(seg), 1))
                for j in range(len(seg)):
                    x_lon[j] = seg[j][0]
                    y_lat[j] = seg[j][1]

                clr = v_cols[i] if isinstance(v_cols, np.ndarray) else v_cols
                if llenar:
                    eje.fill(x_lon, y_lat, color=clr, alpha=alpha)
                else:
                    eje.plot(x_lon, y_lat, color=clr, alpha=alpha)
        fig.colorbar(cpick, ax=eje)

    @staticmethod
    def _resolver_colores(colores=None):
        if colores is None:
            return ['#FF6666', '#FFCC66', '#00CC66']
        elif colores == -1:
            return ['#00CC66', '#FFCC66', '#FF6666']
        elif isinstance(colores, str):
            return ['#FFFFFF', colores]
        return colores


def _hex_a_rva(hx):
    """
    Convierte colores RVA a Hex.

    Parameters
    ----------
    hx: str
        El valor hex.

    Returns
    -------
    tuple
        El valor rva.
    """
    return tuple(int(hx.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))


def _gen_d_mapacolores(colores):
    """
    Genera un diccionario de mapa de color para MatPlotLib.

    Parameters
    ----------
    colores: list
        Una lista de colores

    Returns
    -------
    dict
        Un diccionario para MatPlotLib
    """

    clrs_rva = [_hex_a_rva(x) for x in colores]
    # noinspection PyTypeChecker
    n_colores = len(colores)

    dic_c = {'red': tuple((round(i / (n_colores - 1), 2), clrs_rva[i][0] / 255, clrs_rva[i][0] / 255) for i in
                          range(0, n_colores)),
             'green': tuple(
                 (round(i / (n_colores - 1), 2), clrs_rva[i][1] / 255, clrs_rva[i][1] / 255) for i in
                 range(0, n_colores)),
             'blue': tuple(
                 (round(i / (n_colores - 1), 2), clrs_rva[i][2] / 255, clrs_rva[i][2] / 255) for i in
                 range(0, n_colores))}

    return dic_c
