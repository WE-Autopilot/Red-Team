import os
from weap_util.weap_container import run
from controller import Controller

if __name__ == "__main__":
    controller = Controller(model_path="f110_line_sensor_sac.zip")

    ##maps_dir = os.path.abspath(os.path.join("..", "assets", "maps"))
    map_name = "map0"

    current_directory = os.getcwd()
    maps_dir = os.path.abspath(os.path.join(current_directory, "..", "assets","map0"))
    run(controller, config_path=maps_dir, config_name=map_name, render_on=True)
