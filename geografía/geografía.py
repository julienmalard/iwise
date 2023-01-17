import shapefile as sf


class Geografía(object):
    def __init__(símismo, archivo: str, país: str):
        símismo.forma = sf.Reader(archivo)
