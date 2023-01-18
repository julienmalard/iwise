import os.path

import numpy as np
import pandas as pd

from constantes import COL_REGIÓN, COLS_REGIONES, DIR_EGRESO, COL_PAÍS, COL_SEGGHÍD
from geografía.geografía import Geografía
from modelo import Modelo, ConfigDatos
from traducciones import traducciones

lenguaje = "es"


def preparar_datos():
    datos_pd = pd.read_stata("datos/IWISE2020_LatinAmerica.dta")

    # Combinar columnas de región
    datos_pd[COL_REGIÓN] = np.nan
    for col in COLS_REGIONES:
        datos_pd[COL_REGIÓN] = datos_pd[COL_REGIÓN].fillna(datos_pd[col])

    if lenguaje != "en":
        for palabra, trads in traducciones.items():
            if trads[lenguaje]:
                datos_pd = datos_pd.replace([palabra], trads[lenguaje])

    return datos_pd


def preparar_config():
    datos_pd = preparar_datos()
    return ConfigDatos(datos_pd, dir_egreso=DIR_EGRESO, col_país=COL_PAÍS, col_región=COL_REGIÓN)

# Mapa IARNA
Guatemala = Geografía(
    os.path.join("geografía", "mapas", "guatemala", "departamentos_gtm_fin"),
    país="Guatemala",
    columna_región="FIRST_DEPA",
    args_shp={"encoding": "latin-1"}
)

# https://data.humdata.org/dataset/cod-ab-hnd
Honduras = Geografía(
    os.path.join("geografía", "mapas", "honduras", "hnd_admbnda_adm1_sinit_20161005"),
    país="Honduras",
    columna_región="ADM1_ES",
    traslado_nombres={
        "Atlantida": "Atlántida",
        "Cortes": "Cortés",
        "Islas de La Bahia": "Isla de la Bahía",
        "Santa Barbara": "Santa Bárbara",
        "Copan": "Copán",
        "Intibuca": "Intibucá",
        "Francisco Morazan": "Francisco Morazán",
        "El Paraiso": "El Paraíso"
    }
)

Brazil = Geografía(
    os.path.join("geografía", "mapas", "brazil", "bra_admbnda_adm1_ibge_2020"),
    país="Brazil",
    columna_región="ADM1_PT",
    traslado_nombres={
        "Espírito Santo": "Espirito Santo",
        "Rondônia": "Rondonia"
    }
)


if __name__ == "__main__":
    config = preparar_config()

    modelo = Modelo("Región", var_y=COL_SEGGHÍD, var_x=COL_REGIÓN, config=config).dibujar()
    Brazil.dibujar(modelo, colores=-1)
    Guatemala.dibujar(modelo, colores=-1)
    Honduras.dibujar(modelo, colores=-1)

    Modelo("Género", var_y=COL_SEGGHÍD, var_x="WP1219", config=config).dibujar()

    Modelo("Ruralidad", var_y=COL_SEGGHÍD, var_x="WP14", config=config).dibujar()

    Modelo("Matrimonio", var_y=COL_SEGGHÍD, var_x="WP1223", config=config).dibujar()

    Modelo("Nivel educativo", var_y=COL_SEGGHÍD, var_x="WP3117", config=config).dibujar()

    Modelo("Empleo", var_y=COL_SEGGHÍD, var_x="EMP_2010", config=config).dibujar()

    Modelo("Religión", var_y=COL_SEGGHÍD, var_x="WP1233RECODED", config=config).dibujar()

    Modelo("Dificultad económica", var_y=COL_SEGGHÍD, var_x="WP2319", config=config).dibujar()

    Modelo("Clase económica", var_y=COL_SEGGHÍD, var_x="INCOME_5", config=config).dibujar()
