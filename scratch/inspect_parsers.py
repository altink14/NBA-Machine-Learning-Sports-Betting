import nba_api.stats.endpoints._parsers as parsers
import inspect

source_file = inspect.getsourcefile(parsers)
print("Parsers file path:", source_file)

with open(source_file, "r", encoding="utf-8") as f:
    code = f.read()

# Find class names
class_names = re.findall(r'class\s+([A-Za-z0-9_]+)', code) if 're' in globals() else []
# import re first
import re
class_names = re.findall(r'class\s+([A-Za-z0-9_]+)', code)
print("\nTotal classes in _parsers.py:", len(class_names))

print("\nClasses containing interest keywords:")
for cls in class_names:
    for kw in ["gravity", "leverage", "difficulty", "dunk", "leader"]:
        if kw in cls.lower():
            print(f"  Class: {cls}")
            # Print class definition snippet
            cls_match = re.search(fr'class\s+{cls}.*?(?=\n\s*class|\Z)', code, re.DOTALL)
            if cls_match:
                lines = cls_match.group(0).split('\n')
                print("    " + "\n    ".join(lines[:10]))
            break
