from os import path, makedirs

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm

COL_PAÍS = "countrynew"
DIR_EGR = "resultados"

if not path.isdir(DIR_EGR):
    makedirs(DIR_EGR)


class Modelo(object):
    def __init__(símismo, nombre: str, var_y: str, var_x: str, datos: pd.DataFrame, recalibrar=False):
        símismo.nombre = nombre
        símismo.var_y = var_y
        símismo.var_x = var_x
        símismo.datos = datos[[var_y, var_x, COL_PAÍS]].dropna()
        símismo.recalibrar = recalibrar

    def calibrar(símismo, país: str):
        datos_país = símismo.datos.loc[símismo.datos[COL_PAÍS] == país]
        datos_x = datos_país[símismo.var_x]
        categorías_x = pd.Categorical(datos_x)

        with pm.Model():
            a = pm.Normal(name="a", shape=categorías_x.categories.size)

            índices_a = categorías_x.codes
            pm.Bernoulli(logit_p=a[índices_a], name="prob", observed=datos_país[símismo.var_y])

            pm.Deterministic("b", pm.math.invlogit(a))

            traza = pm.sample()

        az.to_netcdf(traza, símismo.archivo(país))

    def dibujar(símismo):
        países = símismo.datos[COL_PAÍS].unique()

        for país in países:
            if símismo.recalibrar or not path.isfile(símismo.archivo(país)):
                símismo.calibrar(país)

            datos_país = símismo.datos.loc[símismo.datos[COL_PAÍS] == país]
            datos_x = datos_país[símismo.var_x]
            categorías_x = pd.Categorical(datos_x)

            traza = az.from_netcdf(símismo.archivo(país))
            az.plot_trace(traza, ["b"])
            fig = plt.gcf()
            fig.suptitle(f"{país}: Probabilidad por {', '.join(categorías_x.categories.tolist())}")
            fig.savefig(símismo.archivo_gráfico(país))

    def archivo(símismo, país: str):
        return path.join(DIR_EGR, f"{símismo.nombre}-{país}.ncdf")

    def archivo_gráfico(símismo, país: str):
        return path.join(DIR_EGR, f"{símismo.nombre}-{país}.jpg")
