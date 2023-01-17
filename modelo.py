import math
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
        símismo.datos = config.datos[list({var_y, var_x, config.col_país, config.col_región})].dropna()

        símismo.recalibrado = False

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

        az.to_netcdf(traza, símismo.archivo_calibs(país))
        símismo.recalibrado = True

    def dibujar(símismo, recalibrar=False):
        símismo.dibujar_traza(recalibrar)
        símismo.dibujar_caja_bigotes(recalibrar)

        return símismo

    def dibujar_traza(símismo, recalibrar=False):
        países = símismo.datos[símismo.config.col_país].unique()

        for país in países:
            traza = símismo.obt_traza(país, recalibrar)

            datos_país = símismo.datos.loc[símismo.datos[símismo.config.col_país] == país]
            datos_x = datos_país[símismo.var_x]
            categorías_x = pd.Categorical(datos_x)

            az.plot_trace(traza, ["b"])
            fig = plt.gcf()
            fig.suptitle(f"{país}: Probabilidad por {', '.join(categorías_x.categories.tolist())}")
            fig.savefig(símismo.archivo_gráfico(país, "traza"))
            plt.close(fig)

    def dibujar_caja_bigotes(símismo, recalibrar=False):
        países = símismo.datos[símismo.config.col_país].unique()

        for país in países:
            datos_país = símismo.datos.loc[símismo.datos[símismo.config.col_país] == país]
            datos_x = datos_país[símismo.var_x]
            categorías_x = pd.Categorical(datos_x).categories.values.tolist()

            fig, ejes = plt.subplots(1, 2, figsize=(12, 6))
            fig.subplots_adjust(bottom=0.2)

            traza_por_categoría = símismo.obt_traza_por_categoría(país, recalibrar)
            n_categ = len(traza_por_categoría.columns)

            # Dibujar distribución
            dibujo_dist = sns.kdeplot(traza_por_categoría, ax=ejes[0])

            # Ajustar leyenda
            sns.move_legend(
                dibujo_dist, ncols=max(2, math.ceil(n_categ / 3)), bbox_to_anchor=(1.1, -0.17),
                loc="center"
            )

            # Dibujar caja
            caja = traza_por_categoría.boxplot(ax=ejes[1], grid=False, return_type="dict")
            colores_por_categ = {
                categorías_x[i]: dibujo_dist.legend_.legendHandles[i].get_color() for i in range(n_categ)
            }
            ejes[1].set(xticklabels=[])
            for categ, color in colores_por_categ.items():
                i = categorías_x.index(categ)
                for forma in ["boxes", "medians"]:
                    caja[forma][i].set_color(color)
                caja["fliers"][i].set_markeredgecolor(color)
                for forma in ["whiskers", "caps"]:
                    for j in range(2):
                        caja[forma][i * 2 + j].set_color(color)

            fig.suptitle(f"{país}: Probabilidad de inseguridad hídrica por {símismo.nombre.lower()}")
            fig.savefig(símismo.archivo_gráfico(país, "caja"))
            plt.close(fig)

    def obt_traza(símismo, país: str, recalibrar=False):
        if (recalibrar and not símismo.recalibrado) or not path.isfile(símismo.archivo_calibs(país)):
            símismo.calibrar(país)
        return az.from_netcdf(símismo.archivo_calibs(país))

    def obt_traza_por_categoría(símismo, país: str, recalibrar=False) -> pd.DataFrame:
        traza = símismo.obt_traza(país, recalibrar)

        categorías = traza.posterior["b_dim_0"].values

        datos_país = símismo.datos.loc[símismo.datos[símismo.config.col_país] == país]
        datos_x = datos_país[símismo.var_x]
        categorías_x = pd.Categorical(datos_x)
        return pd.DataFrame({
            categorías_x.categories[c]: traza.posterior["b"].sel({"b_dim_0": c}).values.flatten() for c in categorías
        })

    def archivo_calibs(símismo, país: str) -> str:
        dir_calibs = path.join(símismo.config.dir_egreso, "calibs")
        if not path.isdir(dir_calibs):
            makedirs(dir_calibs)
        return path.join(dir_calibs, f"{símismo.nombre}-{país}.ncdf")

    def archivo_gráfico(símismo, país: str, tipo: str) -> str:
        dir_gráfico = path.join(símismo.config.dir_egreso, tipo)
        if not path.isdir(dir_gráfico):
            makedirs(dir_gráfico)

        return path.join(dir_gráfico, f"{símismo.nombre}-{país}.jpg")


class ModeloRegional(Modelo):

    def calibrar(símismo, país: str):
        datos_país = símismo.datos.loc[símismo.datos[símismo.config.col_país] == país]
        datos_x = datos_país[símismo.var_x]
        categorías_x = pd.Categorical(datos_x)

        datos_región = datos_país[símismo.config.col_región]
        regiones = pd.Categorical(datos_región)

        with pm.Model():
            a_país = pm.Normal(name="a_país", shape=categorías_x.categories.size)
            a = pm.Normal(name="a", mu=a_país, shape=(regiones.categories.size, categorías_x.categories.size))

            índices_a = categorías_x.codes
            índices_región = regiones.codes
            pm.Bernoulli(logit_p=a[índices_región, índices_a], name="prob", observed=datos_país[símismo.var_y])

            b = pm.Deterministic("b_por_género", pm.math.invlogit(a))
            pm.Deterministic("b", b[..., 1] - b[..., 0])

            traza = pm.sample()

        az.to_netcdf(traza, símismo.archivo_calibs(país))

    def archivo_calibs(símismo, país: str):
        return super().archivo_calibs(país=f"{país}-regional")

    def archivo_gráfico(símismo, país: str, tipo: str):
        return super().archivo_gráfico(país=f"{país}-regional", tipo=tipo)
