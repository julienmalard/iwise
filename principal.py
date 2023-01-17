import numpy as np
import pandas as pd

from constantes import COL_REGIÓN, COLS_REGIONES, DIR_EGRESO, COL_PAÍS
from modelo import Modelo, ConfigDatos


def preparar_datos():
    datos_pd = pd.read_stata("datos/IWISE2020_LatinAmerica.dta")

    # Combinar columnas de región
    datos_pd[COL_REGIÓN] = np.nan
    for col in COLS_REGIONES:
        datos_pd[COL_REGIÓN] = datos_pd[COL_REGIÓN].fillna(datos_pd[col])

    return datos_pd


def preparar_config():
    datos_pd = preparar_datos()
    return ConfigDatos(datos_pd, dir_egreso=DIR_EGRESO, col_país=COL_PAÍS, col_región=COL_REGIÓN)


if __name__ == "__main__":
    config = preparar_config()

    Modelo("Género", var_y="iwise12", var_x="WP1219", config=config).dibujar()

    Modelo("Ruralidad", var_y="iwise12", var_x="WP14", config=config).dibujar()

    Modelo("Matrimonio", var_y="iwise12", var_x="WP1223", config=config).dibujar()

    Modelo("Nivel educativo", var_y="iwise12", var_x="WP3117", config=config).dibujar()

    Modelo("Empleo", var_y="iwise12", var_x="EMP_2010", config=config).dibujar()

    Modelo("Religión", var_y="iwise12", var_x="WP1233RECODED", config=config).dibujar()

    Modelo("Dificultad económica", var_y="iwise12", var_x="WP2319", config=config).dibujar()

    Modelo("Clase económica", var_y="iwise12", var_x="INCOME_5", config=config).dibujar()
