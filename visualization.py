import polyscope as ps
import numpy as np
import os, random, json

# from data.metadata import ShapeNET_classes

# Inicializa polyscope, creando contextos graficos y construyendo una ventana.
ps.init()

# Carga un archivo de nube de puntos
datasets_folder = os.path.join(
    os.path.dirname(__file__), "data\\ShapeNet55-34\\shapenet_pc"
)
random_points = random.choice(os.listdir(datasets_folder))
points = np.load(os.path.join(datasets_folder, random_points))

cloud = ps.register_point_cloud("my points", points)
cloud_class = random_points[:8]

# Print the class of the cloud
with open("data\\ShapeNet55-34\\Shapenet_classes.json") as json_file:
    ShapeNET_classes = json.load(json_file)
    print(ShapeNET_classes[cloud_class])

ps.show()
