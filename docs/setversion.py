import starry
import re
with open("Doxyfile", "r") as f:
    file = f.read()
file = re.sub('PROJECT_NUMBER\s*?= "(.*?)"\n', 
              'PROJECT_NUMBER = "%s"\n' % starry.__version__, file)
with open("Doxyfile", "w") as f:
    print(file, file=f)