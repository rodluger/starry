import starry
import packaging
import urllib

# Hack to figure out if we are the latest version
url = "https://raw.githubusercontent.com/rodluger/starry/gh-pages/versions.txt"
all_versions = []
for line in urllib.request.urlopen(url):
    version_string = line.decode("utf-8").replace("\n", "").strip()
    all_versions.append(packaging.version.parse(version_string))
all_versions = sorted(all_versions)
current_version = packaging.version.parse(starry.__version__)
is_latest = (current_version.is_devrelease) and (
    current_version.base_version >= all_versions[-1].base_version
)

if is_latest:
    print("latest")
else:
    print("v{}".format(current_version.base_version))
