from packaging import version
import functools
import sys


def cmp(v1, v2):
    if v1 == "latest":
        return 1
    elif v2 == "latest":
        return -1
    elif version.parse(v1) < version.parse(v2):
        return -1
    elif version.parse(v1) > version.parse(v2):
        return 1
    else:
        return 0


# Get all versions
with open("versions.txt", "r") as f:
    versions = [v.split("\n")[0] for v in f.readlines()]

# Add the current version to the list
if len(sys.argv) == 2:
    versions += [sys.argv[1]]

# Eliminate duplicates, if any
versions = list(set(versions))

# Eliminate non-version entries, if any
versions = [
    v
    for v in versions
    if isinstance(version.parse(v), version.Version) or v == "latest"
]

# Sort them
versions = sorted(versions, key=functools.cmp_to_key(cmp))

# Output back to the same file
with open("versions.txt", "w") as f:
    for v in versions[:-1]:
        print(v, file=f)
    print(versions[-1], file=f, end="")
