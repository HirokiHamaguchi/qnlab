import os

problems_dir = os.path.dirname(__file__)

exports = []
for filename in sorted(os.listdir(problems_dir)):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]
        with open(os.path.join(problems_dir, filename)) as f:
            for line in f:
                if line.startswith("class "):
                    class_name = line.split()[1].split("(")[0].split(":")[0]
                    exports.append((module_name, class_name))
                    break

init_file = os.path.join(problems_dir, "__init__.py")
with open(init_file, "w") as f:
    for module, class_name in exports:
        f.write(f"from qnlab.problem.{module} import {class_name}\n")
    f.write("\n__all__ = [\n")
    for _, class_name in exports:
        f.write(f'    "{class_name}",\n')
    f.write("]\n")
