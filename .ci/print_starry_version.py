import starry
import packaging

full_version = packaging.version.parse(starry.__version__)
base_version = full_version.base_version
print("v{}".format(base_version))
