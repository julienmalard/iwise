from os import path, makedirs

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm
import seaborn as sns


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
    def __init__(símismo, nombre: str, var_y: str, var_x: str, config: ConfigDatos):
        símismo.nombre = nombre
        símismo.var_y = var_y
        símismo.var_x = var_x
        símismo.config = config
        símismo.datos = config.datos[[var_y, var_x, config.col_país]].dropna()

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

    def dibujar(símismo, recalibrar=False):
        símismo.dibujar_traza(recalibrar)
        símismo.dibujar_caja_bigotes(recalibrar)

    def dibujar_traza(símismo, recalibrar=False):
        países = símismo.datos[símismo.config.col_país].unique()

        for país in países:
            if recalibrar or not path.isfile(símismo.archivo(país)):
                símismo.calibrar(país)

            datos_país = símismo.datos.loc[símismo.datos[símismo.config.col_país] == país]
            datos_x = datos_país[símismo.var_x]
            categorías_x = pd.Categorical(datos_x)

            traza = az.from_netcdf(símismo.archivo(país))
            az.plot_trace(traza, ["b"])
            fig = plt.gcf()
            fig.suptitle(f"{país}: Probabilidad por {', '.join(categorías_x.categories.tolist())}")
            fig.savefig(símismo.archivo_gráfico(país, "traza"))

    def dibujar_caja_bigotes(símismo, recalibrar=False):
        países = símismo.datos[símismo.config.col_país].unique()

        for país in países:
            if recalibrar or not path.isfile(símismo.archivo(país)):
                símismo.calibrar(país)

            datos_país = símismo.datos.loc[símismo.datos[símismo.config.col_país] == país]
            datos_x = datos_país[símismo.var_x]
            categorías_x = pd.Categorical(datos_x)

            traza = az.from_netcdf(símismo.archivo(país))

            fig, ejes = plt.subplots(1, 2, figsize=(12, 6))

            categorías = traza.posterior["b_dim_0"].values
            traza_por_categoría = pd.DataFrame({
                categorías_x.categories[c]: traza.posterior["b"].sel({"b_dim_0": c}).values.flatten() for c in categorías
            })
            # Dibujar distribución
            dibujo_dist = sns.kdeplot(traza_por_categoría, ax=ejes[0])

            # Dibujar caja
            caja = traza_por_categoría.boxplot(ax=ejes[1], grid=False, return_type="dict")
            colores_por_categ = [(dibujo_dist.legend_.legendHandles[i].get_color(), dibujo_dist.legend_.texts[i].get_text()) for i in categorías]
            for color, categ in colores_por_categ:
                i = categorías_x.categories.values.tolist().index(categ)
                for forma in ["boxes", "medians"]:
                    caja[forma][i].set_color(color)
                caja["fliers"][i].set_markeredgecolor(color)
                for forma in ["whiskers", "caps"]:
                    for j in range(2):
                        caja[forma][i * 2 + j].set_color(color)

            fig.suptitle(f"{país}: Probabilidad por {símismo.nombre.lower()}")
            fig.savefig(símismo.archivo_gráfico(país, "caja"))

    def archivo(símismo, país: str):
        return path.join(símismo.config.dir_egreso, f"{símismo.nombre}-{país}.ncdf")

    def archivo_gráfico(símismo, país: str, tipo: str):
        dir_gráfico = path.join(símismo.config.dir_egreso, tipo)
        if not path.isdir(dir_gráfico):
            makedirs(dir_gráfico)

        return path.join(dir_gráfico, f"{símismo.nombre}-{país}.jpg")
