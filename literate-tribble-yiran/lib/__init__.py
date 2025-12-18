from . import assets, models


def init_dir(app_root_dir):
    models.init_model_dir(f"{app_root_dir}/models")
    assets.init_assets_dir(f"{app_root_dir}/assets")
