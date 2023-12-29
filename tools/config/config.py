
from dynaconf import Dynaconf
import os

work_dir = os.getcwd()
settings_path = os.path.join(work_dir, '..\\config\\settings.yaml')
secrets_path = os.path.join(work_dir, '..\\config\\.secrets.yaml')

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=[settings_path, secrets_path],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
