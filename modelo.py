from os import path

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm


class ConfigDatos(object):
    def __init__(
            símismo, datos: pd.DataFrame,
            dir_egreso: str,
            col_país: str,
            col_región: str
    ):
        símismo.datos = datos
        símismo.dir_egreso = dir_egreso
        símismo.col_país = col_país
        símismo.col_región = col_región


class Modelo(object):
    def __init__(símismo, nombre: str, var_y: str, var_x: str, config: ConfigDatos, recalibrar=False):
        símismo.nombre = nombre
        símismo.var_y = var_y
        símismo.var_x = var_x
        símismo.config = config
        símismo.datos = config.datos[[var_y, var_x, config.col_país]].dropna()
        símismo.recalibrar = recalibrar

    def calibrar(símismo, país: str):
        datos_país = símismo.datos.loc[símismo.datos[símismo.config.col_país] == país]
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
        países = símismo.datos[símismo.config.col_país].unique()

        for país in países:
            if símismo.recalibrar or not path.isfile(símismo.archivo(país)):
                símismo.calibrar(país)

            datos_país = símismo.datos.loc[símismo.datos[símismo.config.col_país] == país]
            datos_x = datos_país[símismo.var_x]
            categorías_x = pd.Categorical(datos_x)

            traza = az.from_netcdf(símismo.archivo(país))
            az.plot_trace(traza, ["b"])
            fig = plt.gcf()
            fig.suptitle(f"{país}: Probabilidad por {', '.join(categorías_x.categories.tolist())}")
            fig.savefig(símismo.archivo_gráfico(país))

    def archivo(símismo, país: str):
        return path.join(símismo.config.dir_egreso, f"{símismo.nombre}-{país}.ncdf")

    def archivo_gráfico(símismo, país: str):
        return path.join(símismo.config.dir_egreso, f"{símismo.nombre}-{país}.jpg")
