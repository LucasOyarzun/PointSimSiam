import polyscope as ps
import numpy as np
import os, random
from data.metadata import ShapeNET_classes

# Opciones de visualización
# ps.set_program_name("important app")
# ps.set_verbosity(0)
# ps.set_use_prefs_file(False)

# Opciones de escena
# ps.set_autocenter_structures(True)
# ps.set_autoscale_structures(True)

# Opciones de cámara
# ps.set_navigation_style("free")
# ps.set_up_dir("z_up")
# ps.look_at((0., 0., 5.), (1., 1., 1.))


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
print(ShapeNET_classes[cloud_class])

# Pasa el control del flujo a polyscope, desplegando la ventana. Retorna cuando se cierra
ps.show()
