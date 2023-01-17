import pandas as pd

from modelo import Modelo

if __name__ == "__main__":
    datos = pd.read_stata("datos/IWISE2020_LatinAmerica.dta")

    Modelo("Género", var_y="iwise12", var_x="WP1219", datos=datos).dibujar()

    Modelo("Ruralidad", var_y="iwise12", var_x="WP14", datos=datos).dibujar()

    Modelo("Matrimonio", var_y="iwise12", var_x="WP1223", datos=datos).dibujar()

    Modelo("Nivel educativo", var_y="iwise12", var_x="WP3117", datos=datos).dibujar()

    Modelo("Empleo", var_y="iwise12", var_x="EMP_2010", datos=datos).dibujar()

    Modelo("Religión", var_y="iwise12", var_x="WP1233RECODED", datos=datos).dibujar()

    Modelo("Dificultad económica", var_y="iwise12", var_x="WP2319", datos=datos).dibujar()

    Modelo("Clase económica", var_y="iwise12", var_x="INCOME_5", datos=datos).dibujar()
